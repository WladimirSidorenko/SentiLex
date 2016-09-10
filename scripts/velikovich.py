#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Velikovich's method (2010).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import lemmatize, POSITIVE, NEGATIVE, TOKENIZER, \
    SYNRELS, ANTIRELS, NEGATORS, STOP_WORDS, FORM2LEMMA, \
    TAB_RE, ENCODING, check_word
from germanet import normalize

from collections import Counter, defaultdict
from itertools import chain
from scipy import sparse
import codecs
import numpy as np
import sys


##################################################################
# Constants
ENCODING = "utf-8"
TOK_WINDOW = 4                  # it actually corresponds to six
MAX_NGHBRS = 25
FASTMODE = True
FMAX = sys.float_info.max
FMIN = -FMAX
DFLT_T = 10


##################################################################
# Methods
def _read_files(a_crp_files):
    """Read corpus files and populate one-directional co-occurrences.

    @param a_crp_files - files of the original corpus

    @return (max_vecid, word2vecid, tok_stat)

    @note constructs statistics in place

    """
    print("Reading corpus...", end="", file=sys.stderr)
    i = 0
    itok = ""
    max_vecid = 0
    prev_toks = []
    tok_stat = Counter()
    word2vecid = {}
    for ifname in a_crp_files:
        with codecs.open(ifname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip().lower()
                if not iline:
                    if FASTMODE:
                        i += 1
                        if i > 300:
                            break
                    del prev_toks[:]
                    continue
                itok = TAB_RE.split(iline)[0].strip()
                if not check_word(itok):
                    continue
                if itok not in word2vecid:
                    word2vecid[itok] = max_vecid
                    max_vecid += 1
                itok = word2vecid[itok]
                for ptok in prev_toks:
                    tok_stat[(ptok, itok)] += 1
                while len(prev_toks) > TOK_WINDOW:
                    prev_toks.pop(0)
                prev_toks.append(itok)
        del prev_toks[:]
    print(" done", file=sys.stderr)
    return (max_vecid, word2vecid, tok_stat)


def _tokstat2mtx(a_max_vecid, a_tok_stat):
    """Construct co-occurrence matrix from token statistics.

    @param a_max_vecid - mximum number of unique tokens in text
    @param a_tok_stat - co-occurrence statistics on tokens

    @return adjacency csr matrix

    """
    # lil is better for construction
    M = sparse.lil_matrix((a_max_vecid, a_max_vecid),
                          dtype=np.float32)
    # for faster matrix construction, we sort the keys
    toks = sorted(a_tok_stat.iterkeys())
    icnt = 0
    for (itok1, itok2) in toks:
        icnt = a_tok_stat[(itok1, itok2)]
        # the counter is mutual, so let's see whether we can go with
        M[itok1, itok2] += icnt
        M[itok2, itok1] += icnt
    # free memory
    a_tok_stat.clear()
    del toks[:]
    # csr is better for computation
    return M.tocsr()


def _prune_vec(a_M, a_i, a_j2dot):
    """Remove all pointers to other vectors except for the top MAX_NGHBRS.

    @param a_M - source matrix to be modified
    @param a_i - index of a_M row which should be changed
    @param a_j2dot - row of a_M which should be changed

    @return \c void

    """
    if len(a_j2dot) > MAX_NGHBRS:
        min_score = sorted(a_j2dot.itervalues(),
                           reverse=True)[MAX_NGHBRS - 1]
        for j, jdot in a_j2dot.iteritems():
            if jdot < min_score:
                a_M[a_i, j] = 0
    a_j2dot.clear()


def _prune_mtx(a_M):
    """Make each row contain at most top MAX_NGHBRS neighbours.

    @param a_M - source matrix to be modified

    @return \c void

    @note modifies `a_M' in place

    """
    idot = min_score = 0.
    j2dot = {}
    irow = None
    prev_i = -1
    for i, j in zip(*a_M.nonzero()):
        if i != prev_i:
            if prev_i >= 0:
                _prune_vec(a_M, prev_i, j2dot)
            irow = a_M.getrow(i).transpose()
            prev_i = i
        # prevent self-loops
        if i == j:
            a_M[i, j] == 0.
            continue
        j2dot[j] = a_M.getrow(j).dot(irow)[0, 0]
    _prune_vec(a_M, prev_i, j2dot)
    a_M.eliminate_zeros()
    a_M.prune()


def _crp2mtx(a_crp_files, a_pos, a_neg):
    """Construct sparse collocation matrix from raw corpus.

    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms

    @return (dict, mtx) - number of tokesn, mapping from tokens to vector ids,
    and adjacency matrix

    """
    # gather one-direction co-occurrence statistics
    max_vecid, word2vecid, tok_stat = _read_files(a_crp_files)
    for w in chain(a_pos, a_neg):
        if w not in word2vecid:
            word2vecid[w] = max_vecid
            max_vecid += 1
    # convert cooccurrence statistics to a sparse matrix
    M = _tokstat2mtx(max_vecid, tok_stat)
    # iterate over the matrix and keep top 25 vectors with the highest cosine
    # similarity
    _prune_mtx(M)
    return (max_vecid, word2vecid, M)


def _p_init(a_N, a_word2vecid, a_seedset, a_seed_val=1.):
    """Construct vector of polarity scores.

    @param a_N - dimension of the con
    @param a_word2vecid - mapping from words to their vector id's
    @param a_seedset - set of seed words
    @param a_seed_val - polarity score for seed terms

    @return (np.arry) - populated array of polarity scores

    """
    ret = np.zeros(a_N)
    for idx in a_seedset:
        ret[idx] = a_seed_val
    return ret


def _velikovich(a_p, a_ids, a_M, a_T):
    """Propagate polarity from one seed set through the entire graph.

    @param a_p - resulting polarity score vector to be modified
    @param a_ids - ids of the seed terms
    @param a_M - adjacency matrix with edge scores
    @param a_T - maximum number of iterations

    @return \c void
    @note modifies `a_p` in place

    """
    krow = None
    sset = set()
    nset = set()
    alpha = defaultdict(float)

    for i in a_ids:
        sset.add(i)
        print("seed term: {:d}".format(i),
              end="\r", file=sys.stderr)
        for t in xrange(a_T):
            print("iteration: {:d}".format(t),
                  end="\r", file=sys.stderr)
            for k in sset:
                krow = a_M.getrow(k)
                for _, j in zip(*krow.nonzero()):
                    alpha[(i, j)] = max(alpha[(i, j)],
                                        (alpha[(i, k)] or 1.) * krow[0, j]
                                        )
                    nset.add(j)
            sset |= nset
            nset.clear()
        sset.clear()

    N = a_M.shape[0]
    for j in xrange(N):
        a_p[j] = sum(alpha[i, j] for i in a_ids)


def velikovich(a_N, a_T, a_crp_files, a_pos, a_neg, a_neut):
    """Method for generating sentiment lexicons using Velikovich's approach.

    @param a_N - number of terms to extract
    @param a_T - maximum number of iterations
    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded

    @return list of terms sorted according to their polarities

    """
    max_vecid, word2vecid, M = _crp2mtx(a_crp_files, a_pos, a_neg)

    pos_ids = set(word2vecid[w] for w in a_pos)
    neg_ids = set(word2vecid[w] for w in a_neg)

    p_pos = _p_init(max_vecid, word2vecid, pos_ids)
    p_neg = _p_init(max_vecid, word2vecid, neg_ids)

    # preform propagation for single sets
    _velikovich(p_pos, pos_ids, M, a_T)
    _velikovich(p_neg, neg_ids, M, a_T)

    beta = float(p_pos.sum()) / float(p_neg.sum() or 1.)

    ret = []
    w_score = 0.
    for w, w_id in word2vecid.iteritems():
        if w_id in pos_ids:
            w_score = FMAX
        elif w_id in neg_ids:
            w_score = FMIN
        else:
            w_score = p_pos[w_id] - beta * p_neg[w_id]
        ret.append((w,
                    POSITIVE if w_score > 0. else NEGATIVE,
                    w_score))
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    if a_N >= 0:
        del ret[:a_N]
    return ret
