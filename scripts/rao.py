#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Rao and Ravichandran's method (2009).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from blair_goldensohn import build_mtx
from common import POSITIVE, NEGATIVE, NEUTRAL
from graph import Graph
from scipy import sparse

import numpy as np
import sys

##################################################################
# Constants
POS_IDX = 0
NEG_IDX = 1
NEUT_IDX = 2
MAX_I = 300


##################################################################
# Methods
def _sign_normalize(a_Y, a_terms2idx, a_pos, a_neg, a_neut,
                    a_set_dflt):
    """Fix seed values and row-normalize the class matrix.

    @param a_Y - class matrix to be changed
    @param a_terms2idx - mapping from term to matrix index
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_set_dflt - function to set the default value of an unkown term

    @return void

    @note modifies the input matrix in place

    """
    for iterm, i in a_terms2idx.iteritems():
        if iterm in a_pos:
            a_Y[i, :] = 0.
            a_Y[i, POS_IDX] = 1.
        elif iterm in a_neg:
            a_Y[i, :] = 0.
            a_Y[i, NEG_IDX] = 1.
        elif iterm in a_neut:
            a_Y[i, :] = 0.
            a_Y[i, NEUT_IDX] = 1.
        else:
            a_set_dflt(a_Y, i)
    # normalize all incident transitions
    Z = a_Y.sum(1)
    nonzero_xy = a_Y.nonzero()
    for i, j in zip(*nonzero_xy):
        a_Y[i, j] /= Z[i, 0] or 1.


def prune_normalize(a_M):
    """Make each of the adjacency matrix sum up to one.

    Args:
      a_M (scipy.sparse.csr): matrix to be normalized

    Returns:
      void:

    Note:
      modifies the input matrix in place

    """
    # remove negative transitions
    nonzero_xy = a_M.nonzero()
    for i, j in zip(*nonzero_xy):
        if a_M[i, j] < 0.:
            a_M[i, j] = 0.
    a_M.prune()
    # normalize all outgoing transitions
    Z = a_M.sum(0)
    nonzero_xy = a_M.nonzero()
    for i, j in zip(*nonzero_xy):
        a_M[i, j] /= Z[0, i]


def rao_min_cut(a_germanet, a_pos, a_neg, a_neut, a_seed_pos,
                a_ext_syn_rels):
    """Extend sentiment lexicons using the min-cut method of Rao (2009).

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no
      restriction)
    @param a_ext_syn_rels - use extended set of synonymous relations

    @return list of polar terms, their polarities, and scores

    """
    sgraph = Graph(a_germanet, a_ext_syn_rels)
    # partition the graph into subjective and objective terms
    mcs, cut_edges, _, _ = sgraph.min_cut(a_pos | a_neg, a_neut, a_seed_pos)
    print("min_cut_score (subj. vs. obj.) = {:d}".format(mcs),
          file=sys.stderr)
    # remove edges belonging to the min cut (i.e., cut the graph)
    for isrc, itrg in cut_edges:
        sgraph.nodes[isrc].pop(itrg, None)
    # separate the graph into positive and negative terms
    mcs, _, pos, neg = sgraph.min_cut(a_pos, a_neg, a_seed_pos)
    print("min_cut_score (pos. vs. neg.) = {:d}".format(mcs),
          file=sys.stderr)
    ret = [(inode[0], POSITIVE, 1.) for inode in pos]
    ret.extend((inode[0], NEGATIVE, -1.) for inode in neg)
    return ret


def rao_lbl_prop(a_germanet, a_pos, a_neg, a_neut, a_seed_pos,
                 a_ext_syn_rels):
    """Extend sentiment lexicons using the lbl-prop method of Rao (2009).

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no
      restriction)
    @param a_ext_syn_rels - use extended set of synonymous relations

    @return list of polar terms, their polarities, and scores

    """
    # obtain and row-normalize the adjacency matrix
    terms = set((ilex, ipos)
                for isynid, ipos in a_germanet.synid2pos.iteritems()
                for ilexid in a_germanet.synid2lexids[isynid]
                for ilex in a_germanet.lexid2lex[ilexid]
                )
    terms2idx = {iterm: i for i, iterm in enumerate(terms)}
    M = build_mtx(a_germanet, terms2idx, set(),
                  a_ext_syn_rels, len(terms))
    prune_normalize(M)
    M = M.transpose()
    # check that the matrix is column normalized
    assert np.all(i == 0 or np.isclose([i], [1.])
                  for i in M.sum(0)[0, :])
    # initialize label matrix
    cls2idx = {POSITIVE: POS_IDX, NEGATIVE: NEG_IDX, NEUTRAL: NEUT_IDX}
    Y = sparse.lil_matrix((len(terms), len(cls2idx)), dtype=np.float32)

    def _set_neut_one(X, i):
        X[i, NEUT_IDX] = 1.

    _sign_normalize(Y, terms2idx, a_pos, a_neg, a_neut,
                    _set_neut_one)
    Y = Y.tocsr()

    # perform multiplication until convergence
    def _row_norm(X, i):
        pass

    i = 0
    prev_Y = None
    while prev_Y != Y and i < MAX_I:
        print("i =", repr(i), file=sys.stderr)
        prev_Y = Y.copy()
        Y = M.dot(Y)
        _sign_normalize(Y, terms2idx, a_pos, a_neg, a_neut,
                        _row_norm)
        i += 1
    print("Y =", repr(Y), file=sys.stderr)
    sys.exit(66)
