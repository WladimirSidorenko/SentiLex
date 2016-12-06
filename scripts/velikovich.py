#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Velikovich's method (2010).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import ENCODING, ESC_CHAR, FMAX, FMIN, \
    INFORMATIVE_TAGS, NEGATIVE, POSITIVE, SENT_END_RE, \
    TAB_RE, NONMATCH_RE, MIN_TOK_CNT, check_word
from germanet import normalize

from collections import Counter, defaultdict
from itertools import chain
from scipy import sparse
import codecs
import numpy as np
import sys


##################################################################
# Constants
DFLT_T = 20
FASTMODE = False
MAX_NGHBRS = 25
TOK_WINDOW = 4                  # it actually corresponds to a window of six
MAX_POS_IDS = 10000


##################################################################
# Methods
def _read_files(a_crp_files, a_pos, a_neg,
                a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE):
    """Read corpus files and populate one-directional co-occurrences.

    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return (max_vecid, word2vecid, tok_stat)

    @note constructs statistics in place

    """
    print("Reading corpus...", end="", file=sys.stderr)
    i = 0
    prev_lemmas = []
    tok_stat = Counter()
    word2cnt = Counter()
    iform = itag = ilemma = ""
    for ifname in a_crp_files:
        with codecs.open(ifname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip().lower()
                if not iline or SENT_END_RE.match(iline) \
                   or iline[0] == ESC_CHAR:
                    if FASTMODE and prev_lemmas:
                        i += 1
                        if i > 300:
                            break
                    if prev_lemmas:
                        del prev_lemmas[:]
                    continue
                try:
                    iform, itag, ilemma = TAB_RE.split(iline)
                except:
                    print("Invalid line format at line: {:s}".format(
                        repr(iline)), file=sys.stderr
                    )
                    continue
                ilemma = normalize(ilemma)
                if a_pos_re.search(iform) or a_neg_re.search(iform) \
                   or a_pos_re.search(ilemma) or a_neg_re.search(ilemma):
                    pass
                elif itag[:2] not in INFORMATIVE_TAGS \
                        or not check_word(ilemma):
                    continue
                word2cnt[ilemma] += 1
                for plemma in prev_lemmas:
                    tok_stat[(plemma, ilemma)] += 1
                while len(prev_lemmas) > TOK_WINDOW:
                    prev_lemmas.pop(0)
                prev_lemmas.append(ilemma)
        del prev_lemmas[:]
    print(" done", file=sys.stderr)
    max_vecid = 0
    word2vecid = {}
    # convert words to vector ids if their counters are big enough
    for w, cnt in word2cnt.iteritems():
        if cnt >= MIN_TOK_CNT or w in a_pos or w in a_neg:
            word2vecid[w] = max_vecid
            max_vecid += 1
    word2cnt.clear()
    # convert words to vector ids in context counter
    tok_stat = {(word2vecid[w1], word2vecid[w2]): cnt
                for (w1, w2), cnt in tok_stat.iteritems()
                if w1 in word2vecid and w2 in word2vecid
                and cnt >= MIN_TOK_CNT
                }
    return (max_vecid, word2vecid, tok_stat)


def _tokstat2mtx(a_max_vecid, a_tok_stat):
    """Construct co-occurrence matrix from token statistics.

    @param a_max_vecid - mximum number of unique tokens in text
    @param a_tok_stat - co-occurrence statistics on tokens

    @return adjacency csr matrix

    """
    # lil is better for construction
    M = sparse.lil_matrix((a_max_vecid, a_max_vecid),
                          dtype=np.float64)
    # for faster matrix construction, we sort the keys
    toks = sorted(a_tok_stat.iterkeys())
    icnt = 0
    for (itok1, itok2) in toks:
        icnt = a_tok_stat[(itok1, itok2)]
        # the counter is mutual, so let's see whether we can go with
        M[itok1, itok2] += icnt
        assert np.isfinite(M[itok1, itok2]), \
            "Numerical overflow for matrix cell [{:d}, {:d}]".format(
                itok1, itok2)
        M[itok2, itok1] += icnt
        assert np.isfinite(M[itok2, itok1]), \
            "Numerical overflow for matrix cell [{:d}, {:d}]".format(
                itok2, itok1)
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
    prev_i = -1
    irow = None
    ij2dot = defaultdict(dict)
    # prevent self-loops
    for i, j in zip(*a_M.nonzero()):
        if i == j:
            a_M[i, j] == 0.
    # first determine the top-25 most similar words for each vector
    for i, j in zip(*a_M.nonzero()):
        if i != prev_i:
            irow = a_M.getrow(i).transpose()
            prev_i = i
        ij2dot[i][j] = a_M.getrow(j).dot(irow)[0, 0]
        assert np.isfinite(ij2dot[i][j]), \
            "Numerical overflow in dot product of rows {:d} and {:d}".format(
                i, j)
    # actually prune the matrix
    for i in xrange(a_M.shape[0]):
        _prune_vec(a_M, i, ij2dot[i])
    a_M.eliminate_zeros()
    a_M.prune()


def _crp2mtx(a_crp_files, a_pos, a_neg,
             a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE):
    """Construct sparse collocation matrix from raw corpus.

    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return (dict, mtx) - number of tokens, mapping from tokens to vector ids,
    and adjacency matrix

    """
    # gather one-direction co-occurrence statistics
    max_vecid, word2vecid, tok_stat = _read_files(a_crp_files, a_pos, a_neg,
                                                  a_pos_re, a_neg_re)
    for w in chain(a_pos, a_neg):
        w = normalize(w)
        if w not in word2vecid:
            word2vecid[w] = max_vecid
            max_vecid += 1
    # convert cooccurrence statistics to a sparse matrix
    M = _tokstat2mtx(max_vecid, tok_stat)
    # iterate over the matrix and keep top 25 vectors with the highest cosine
    # similarity
    _prune_mtx(M)
    return (max_vecid, word2vecid, M.log1p())


def _p_init(a_N, a_seedset, a_seed_val=1.):
    """Construct vector of polarity scores.

    @param a_N - vocabulary size
    @param a_seedset - set of seed words
    @param a_seed_val - polarity score for seed terms

    @return (np.array) - populated array of polarity scores

    """
    ret = np.zeros(a_N)
    for idx in a_seedset:
        ret[idx] = a_seed_val
    return ret


def _velikovich(a_p, a_ids, a_M, a_T):
    """Propagate polarity score from one seed set through the entire graph.

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
        for t in xrange(a_T):
            for k in sset:
                krow = a_M.getrow(k)
                for _, j in zip(*krow.nonzero()):
                    alpha[(i, j)] = max(alpha[(i, j)],
                                        alpha[(i, k)] + krow[0, j]
                                        )
                    assert np.isfinite(alpha[(i, j)]), \
                        "Numerical overflow occurred when computing" \
                        " alpha[{:d}, {:d}]".format(i, j)
                    nset.add(j)
            sset |= nset
            nset.clear()
        sset.clear()

    for j in xrange(a_M.shape[0]):
        a_p[j] = sum(alpha[i, j] for i in a_ids)
        assert np.isfinite(a_p[j]), \
            "Numerical overflow occurred when computing" \
            " a_p[{:d}]".format(j)


def velikovich(a_N, a_T, a_crp_files, a_pos, a_neg,
               a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE):
    """Method for generating sentiment lexicons using Velikovich's approach.

    @param a_N - number of terms to extract
    @param a_T - maximum number of iterations
    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return list of terms sorted according to their polarities

    """
    max_vecid, word2vecid, M = _crp2mtx(a_crp_files, a_pos, a_neg,
                                        a_pos_re, a_neg_re)

    pos_ids = set(word2vecid[w] for w in a_pos)
    if a_pos_re != NONMATCH_RE:
        add_ids = set(i for w, i in word2vecid.iteritems()
                      if a_pos_re.search(w))
        if len(add_ids) > MAX_POS_IDS:
            add_ids = set(list(add_ids)[:MAX_POS_IDS])
            pos_ids |= add_ids
    neg_ids = set(word2vecid[w] for w in a_neg)
    if a_neg_re != NONMATCH_RE:
        add_ids = set(i for w, i in word2vecid.iteritems()
                      if a_neg_re.search(w))
        if len(add_ids) > MAX_POS_IDS:
            add_ids = set(list(add_ids)[:MAX_POS_IDS])
            pos_ids |= add_ids

    p_pos = _p_init(max_vecid, pos_ids)
    p_neg = _p_init(max_vecid, neg_ids)

    # preform propagation for single sets
    _velikovich(p_pos, pos_ids, M, a_T)
    _velikovich(p_neg, neg_ids, M, a_T)

    beta = float(p_pos.sum()) / float(p_neg.sum() or 1.)

    ret = []
    w_score = 0.
    for w, w_id in word2vecid.iteritems():
        if w in a_pos or a_pos_re.search(w):
            w_score = FMAX
        elif w in a_neg or a_neg_re.search(w):
            w_score = FMIN
        else:
            w_score = p_pos[w_id] - beta * p_neg[w_id]
        ret.append((w,
                    POSITIVE if w_score > 0. else NEGATIVE,
                    w_score))
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    if a_N >= 0:
        del ret[a_N:]
    return ret
