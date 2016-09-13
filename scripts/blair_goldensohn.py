#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Blair-Goldensohn's method (2008).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import ANTIRELS, SYNRELS, POSITIVE, NEGATIVE, NEUTRAL

from itertools import chain
from scipy import sparse

import numpy as np
import sys


##################################################################
# Constants
LAMBDA = 0.2
MAX_ITERS = 5
THRSHLD = 0.5


##################################################################
# Methods
def seeds2seedpos(a_seeds, a_pos):
    """Convert set of seed terms to a set with seed terms and PoS.

    @param a_seeds - set of seed terms
    @param a_pos - list of part-of-speech tags of the seed terms

    @return set of seed terms with their PoS

    """
    return set((iterm, ipos)
               for iterm in a_seeds
               for ipos in a_pos)


def _sign_correct(a_v, a_terms2idx, a_seeds, a_val):
    """Fix polarity values of the seed terms.

    @param a_v - vector of word polarity scores
    @param a_terms2idx - mapping from terms to ``a_v`` indices
    @param a_seeds - set of seed terms
    @param a_val - value used for the seed terms

    """
    idx = 0
    ival = 0.
    for iterm in a_seeds:
        if iterm in a_terms2idx:
            idx = a_terms2idx[iterm]
            ival = a_v[a_terms2idx[iterm]]
            if (ival == 0. and a_val != 0) \
               or (ival != 0. and a_val == 0):
                a_v[idx] = a_val
            elif a_val < 0:
                a_v[idx] = min(ival, a_val)
            elif a_val > 0:
                a_v[idx] = max(ival, a_val)


def sign_correct(a_v, a_terms2idx, a_pos, a_neg, a_neut):
    """Fix polarity values of the seed terms in the given vector.

    @param a_v - vector of word polarity scores
    @param a_terms2idx - mapping from terms to ``a_v`` indices
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_val - value used for the seed terms

    """
    for iseeds, ival in zip((a_pos, a_neg, a_neut),
                            (1., -1., 0.)):
        _sign_correct(a_v, a_terms2idx, iseeds, ival)


def _add_lex_rels(a_M, a_j, a_lexid, a_germanet, a_term2idx, a_neut):
    """Add antonymous relations from GermaNet to the matrix.

    @param a_M - target matrix to update
    @param a_j - index of the source term
    @param a_lexid - id of th lexeme whose antonyms should be added
    @param a_germanet - GermaNet instance
    @param a_term2idx - mapping from term to mtx index
    @param a_neut - set of lexemes with neutral polarity

    @return \c void

    @note modifies ``a_M`` in place

    """
    i = 0
    ival = 0.
    ipos = None
    inode = None
    for ito, irel in a_germanet.lex_relations[a_lexid]:
        if irel in ANTIRELS:
            ival = -LAMBDA
        elif irel in SYNRELS:
            ival = LAMBDA
        else:
            continue
        ipos = None
        for isynid in a_germanet.lexid2synids[ito]:
            if ipos is None:
                ipos = a_germanet.synid2pos[isynid]
            else:
                assert ipos == a_germanet.synid2pos[isynid]
        for ilex in a_germanet.lexid2lex[ito]:
            inode = (ilex, ipos)
            if inode in a_neut:
                continue
            i = a_term2idx[inode]
            a_M[i, a_j] = ival


def _get_con_rels(a_germanet, a_syn_id, a_term2idx):
    """Retrieve con relations pertaining to a synset.

    @param a_germanet - GermaNet instance
    @param a_syn_id - id of the synset whose relations should be retrieved
    @param a_term2idx - mapping from term to mtx index

    @return list of two tuples with the source node indices and values

    """
    # for synonymous ext rels, we need to add an edge to the cell (j, i) due to
    # the asymmetricity of `has_hyponym`, all the remaining edges go to the
    # cell (i, j)
    ret = []
    ival = 0.
    ipos = inode = None
    for ito, irel in a_germanet.con_relations[a_syn_id]:
        if irel in SYNRELS:
            ival = LAMBDA
        elif irel in ANTIRELS:
            ival = -LAMBDA
        else:
            continue
        ipos = a_germanet.synid2pos[ito]
        for ilexid in a_germanet.synid2lexids[ito]:
            for ilex in a_germanet.lexid2lex[ilexid]:
                ret.append((a_term2idx[(ilex, ipos)], ival))
    return ret


def build_mtx(a_germanet, a_term2idx, a_neut, a_ext_rels, a_nterms):
    """Construct adjacency matrix of GermaNet terms.

    @param a_germanet - GermaNet instance
    @param a_term2idx - mapping from term to mtx index
    @param a_neut - set of lexemes with neutral polarity
    @param a_ext_rels - use extended set of synonymous relations
    @param a_nterms - total number of GermaNet terms

    @return adjacency matrix of GermaNet terms

    """
    M = sparse.lil_matrix((a_nterms, a_nterms), dtype=np.float32)
    # set diagonal elements
    lmbda = 1. + LAMBDA
    for i in xrange(a_nterms):
        M[i, i] = lmbda

    val = 0.
    i = j = 0
    ipos = ""
    isrc_nodes = []
    inode = itrg_nodes = None
    for isynid, ilexids in a_germanet.synid2lexids.iteritems():
        ipos = a_germanet.synid2pos[isynid]
        itrg_nodes = [a_term2idx[(ilex, ipos)]
                      for ilexid in ilexids
                      for ilex in a_germanet.lexid2lex[ilexid]]
        if a_ext_rels:
            isrc_nodes = _get_con_rels(a_germanet, isynid, a_term2idx)
        for ilexid in ilexids:
            for ilex in a_germanet.lexid2lex[ilexid]:
                inode = (ilex, ipos)
                if inode in a_neut:
                    continue
                i = a_term2idx[inode]
                # add lex relations from the GermaNet
                _add_lex_rels(M, i, ilexid, a_germanet, a_term2idx,
                              a_neut)
                # add synonymous relations from the synset
                for j in itrg_nodes:
                    if i != j:
                        M[i, j] = LAMBDA
                # add con relations from the GermaNet (it's intended to reverse
                # `from' and `to' here)
                for j, jval in isrc_nodes:
                    if i != j:
                        M[j, i] = jval
    return M.tocsr()


def _vec2pollist(a_terms2vidx, a_v, a_pos, a_neg, a_neut):
    """Convert score vector to a list of polar terms.

    @param a_terms2vidx - mapping from terms to indices in the polar term
      vector
    @param a_v - vector of polar scores
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity

    """
    ret = []
    ret_idx = 0
    lex2lidx = {}
    iscore = jscore = 0.
    ilex = ipos = ipol = ""
    imax, imin = a_v.max() + 1, a_v.min() - 1
    for iterm, ivdx in a_terms2vidx.iteritems():
        (ilex, ipos) = iterm
        iscore = a_v[ivdx]
        # let seed terms have the maximum possible scores
        if iterm in a_pos:
            iscore = imax
        elif iterm in a_neg:
            iscore = imin
        if iterm in a_neut or iscore == 0. or abs(iscore) < THRSHLD:
            continue
        # leave terms whose abs values are greater than threshold
        # determine polarity class
        if iscore > 0.:
            ipol = POSITIVE
        else:
            ipol = NEGATIVE
        # only replace an existing entry, if the new absolute score is higher
        if ilex in lex2lidx:
            ret_idx = lex2lidx[ilex]
            if abs(ret[ret_idx][-1]) < abs(iscore):
                ret[ret_idx] = (ilex, ipol, iscore)
        else:
            lex2lidx[ilex] = len(ret)
            ret.append((ilex, ipol, iscore))
    return ret


def _blair_goldensohn(a_germanet, a_pos, a_neg, a_neut,
                      a_ext_syn_rels):
    """Construct vector and matrix of polar GermaNet terms.

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_ext_syn_rels - use extended set of synonymous relations

    @return set of polar terms, their polarities, and scores

    """
    terms = set((ilex, ipos)
                for isynid, ipos in a_germanet.synid2pos.iteritems()
                for ilexid in a_germanet.synid2lexids[isynid]
                for ilex in a_germanet.lexid2lex[ilexid]
                )
    assert ("kilometerlang", "adj") in terms
    assert ("zugverkehr", "nomen") in terms
    assert ("bewirken", "verben") in terms
    terms2idx = {iterm: i for i, iterm in enumerate(terms)}
    assert ("kilometerlang", "adj") in terms2idx
    assert ("zugverkehr", "nomen") in terms2idx
    assert ("bewirken", "verben") in terms2idx
    # construct term vector
    v = sparse.lil_matrix((len(terms), 1), dtype=np.float32)
    assert v.shape == ((len(terms), 1))
    sign_correct(v, terms2idx, a_pos, a_neg, a_neut)
    assert np.all(v[terms2idx[iterm]] > 0.
                  for iterm in a_pos)
    assert np.all(v[terms2idx[iterm]] < 0.
                  for iterm in a_neg)
    assert np.all(np.isclose(v[terms2idx[iterm]], 0.)
                  for iterm in a_neut)
    # convert term vector to csc_matrix for a more efficient arithmetic
    v = v.tocsc()
    # build adjacency matrix
    M = build_mtx(a_germanet, terms2idx, a_neut,
                  a_ext_syn_rels, len(terms))
    i = np.random.randint(0, len(terms))
    assert np.isclose(M[i, i], [1. + LAMBDA])
    i = terms2idx[("negativ", "adj")]
    j = terms2idx[("positiv", "adj")]
    assert np.isclose(M[i, j], [-LAMBDA])
    assert np.isclose(M[j, i], [-LAMBDA])
    j = terms2idx[("schlecht", "adj")]
    assert np.isclose(M[j, i], [LAMBDA])
    assert np.isclose(M[i, j], [LAMBDA])
    if a_ext_syn_rels:
        j, i = i, terms2idx[("unglücklich", "adj")]
        assert np.isclose(M[i, j], [LAMBDA])
        assert not np.isclose(M[j, i], [LAMBDA])
        i, j = terms2idx[("unförmig", "adj")], \
            terms2idx[("form", "nomen")]
        assert M[j, i] == 0.
        i, j = terms2idx[("lohnend", "adj")], \
            terms2idx[("lohnen", "verben")]
        assert np.isclose(M[j, i], [LAMBDA])
    # propagate the polarity values
    test_val = test_val_prev = 0.
    for _ in xrange(MAX_ITERS):
        test_val_prev = v[terms2idx[("übel", "adj")]]
        v = M.dot(v)
        test_val = v[terms2idx[("übel", "adj")]]
        assert test_val_prev > test_val
        sign_correct(v, terms2idx, a_pos, a_neg, a_neut)
    assert np.all(v[terms2idx[iterm]] > 0.
                  for iterm in a_pos)
    assert np.all(v[terms2idx[iterm]] < 0.
                  for iterm in a_neg)
    assert np.all(np.isclose(v[terms2idx[iterm]], 0.)
                  for iterm in a_neut)
    # free the matrix
    del M
    v = v.toarray()
    return _vec2pollist(terms2idx, v, a_pos, a_neg, a_neut)


def blair_goldensohn(a_germanet, a_pos, a_neg, a_neut,
                     a_seed_pos, a_ext_syn_rels):
    """Extend sentiment lexicons using the  method of Blair-Goldensohn (2010).

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no
      restriction)
    @param a_ext_syn_rels - use extended set of synonymous relations

    @return list of polar terms, their polarities, and scores

    """
    if a_seed_pos == "none":
        a_seed_pos = ["adj", "nomen", "verben"]
    else:
        a_seed_pos = [a_seed_pos]
    a_pos = seeds2seedpos(a_pos, a_seed_pos)
    a_neg = seeds2seedpos(a_neg, a_seed_pos)
    a_neut = seeds2seedpos(a_neut, a_seed_pos)
    # expand seed sets
    ret = _blair_goldensohn(a_germanet, a_pos, a_neg, a_neut,
                            a_ext_syn_rels)
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    return ret
