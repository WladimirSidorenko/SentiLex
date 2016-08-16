#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Rao and Ravichandran's method (2009).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from blair_goldensohn import build_mtx, seeds2seedpos
from common import POSITIVE, NEGATIVE, NEUTRAL
from graph import Graph

from itertools import chain
from scipy import sparse

import numpy as np
import sys

##################################################################
# Constants
POS_IDX = 0
NEG_IDX = 1
NEUT_IDX = 2
POL_IDX = 1
SCORE_IDX = 2
MAX_I = 300
IDX2CLS = {POS_IDX: POSITIVE, NEG_IDX: NEGATIVE, NEUT_IDX: NEUTRAL}


##################################################################
# Methods
def _eq_sparse(a_M1, a_M2):
    """Compare two sparse matrices.

    @param a_M1 - first sparse matrix to compare
    @param a_M2 - second sparse matrix to compare

    @return True if both matrices are equal, non-False otherwise

    """
    if type(a_M1) != type(a_M2):
        return False

    if not np.allclose(a_M1.get_shape(), a_M1.get_shape()):
        return False

    X, Y = a_M1.nonzero()
    IDX1 = set([(x, y) for x, y in zip(X, Y)])
    X, Y = a_M2.nonzero()
    IDX2 = [(x, y) for x, y in zip(X, Y) if (x, y) not in IDX1]
    IDX = list(IDX1)
    IDX.extend(IDX2)
    IDX.sort()

    for x_i, y_i in IDX:
        # print("a_M1[{:d}, {:d}] = {:f}".format(x_i, y_i, a_M1[x_i, y_i]))
        # print("a_M2[{:d}, {:d}] = {:f}".format(x_i, y_i, a_M2[x_i, y_i]))
        # print("is_close", np.isclose(a_M1[x_i, y_i], a_M2[x_i, y_i]))
        if not np.isclose(a_M1[x_i, y_i], a_M2[x_i, y_i]):
            return False
    return True


def _mtx2tlist(a_Y, a_term2idx):
    """Convert matrix to a list of polar terms.

    @param a_Y - matrix of polar terms
    @param a_terms2idx - mapping from terms to their matrix indices

    @return list of 3-tuples (word, polarity, score)

    """
    ret = []
    iscore = 0.
    irow = None
    lex2lidx = {}
    ipol = lidx = 0
    for (iword, ipos), idx in a_term2idx.iteritems():
        # obtain matrix row for that term
        irow = a_Y.getrow(idx).toarray()
        # print("irow =", repr(irow))
        ipol = irow.argmax(axis=1)[0]
        iscore = irow[0, ipol]
        # print("ipol =", repr(ipol))
        # print("iscore =", repr(iscore))
        if ipol != NEUT_IDX:
            ipol = IDX2CLS[ipol]
            if iword in lex2lidx:
                lidx = lex2lidx[iword]
                if abs(iscore) > abs(ret[lidx][SCORE_IDX]):
                    ret[lidx][POL_IDX] = ipol
                    ret[lidx][SCORE_IDX] = iscore
            else:
                lex2lidx[iword] = len(ret)
                ret.append((iword, ipol, iscore))
    return ret


def _sign_normalize(a_Y, a_terms2idx, a_pos, a_neg, a_neut,
                    a_set_dflt=None):
    """Fix seed values and row-normalize the class matrix.

    @param a_Y - class matrix to be changed
    @param a_terms2idx - mapping from terms to their matrix indices
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_set_dflt - function to set the default value of an unkown term

    @return void

    @note modifies the input matrix in place

    """
    seed_found = False
    for iterm, i in a_terms2idx.iteritems():
        if iterm in a_pos:
            seed_found = True
            a_Y[i, :] = 0.
            a_Y[i, POS_IDX] = 1.
        elif iterm in a_neg:
            seed_found = True
            a_Y[i, :] = 0.
            a_Y[i, NEG_IDX] = 1.
        elif iterm in a_neut:
            seed_found = True
            a_Y[i, :] = 0.
            a_Y[i, NEUT_IDX] = 1.
        elif a_set_dflt is not None:
            a_set_dflt(a_Y, i)

    assert seed_found, "No seed term found in matrix."
    # normalize class scores
    Z = a_Y.sum(1)
    x, y = a_Y.nonzero()
    for i, j in zip(x, y):
        # print("a_Y[{:d}, {:d}] =".format(i, j), repr(a_Y[i, j]))
        # print("Z[{:d}, 0] =".format(i), repr(Z[i, 0]))
        a_Y[i, j] /= float(Z[i, 0]) or 1.
        # print("*a_Y[{:d}, {:d}] =".format(i, j), repr(a_Y[i, j]))


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
        a_M[i, j] /= float(Z[0, j]) or 1.


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
        if isrc in sgraph.nodes:
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
    if a_seed_pos == "none":
        a_seed_pos = ["adj", "nomen", "verben"]
    else:
        a_seed_pos = [a_seed_pos]
    a_pos = seeds2seedpos(a_pos, a_seed_pos)
    a_neg = seeds2seedpos(a_neg, a_seed_pos)
    a_neut = seeds2seedpos(a_neut, a_seed_pos)
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

    # no need to transpose M[i, j] is the link going from node j to the node i;
    # and, in Y, the Y[j, k] cell is the polarity score of the class k for the
    # term j
    # M = M.transpose()

    # check that the matrix is column normalized
    assert np.all(i == 0 or np.isclose([i], [1.])
                  for i in M.sum(0)[0, :])
    # initialize label matrix
    Y = sparse.lil_matrix((len(terms), len(IDX2CLS)), dtype=np.float32)

    def _set_neut_one(X, i):
        X[i, NEUT_IDX] = 1.

    _sign_normalize(Y, terms2idx, a_pos, a_neg, a_neut,
                    _set_neut_one)
    # Y = Y.tocsr()

    # output first M row and Y column
    # for i in xrange(len(terms)):
    #     if M[0, i] != 0:
    #         print("M[0, {:d}] =".format(i), M[0, i], file=sys.stderr)
    #     if Y[i, 0] != 0:
    #         print("Y[i, 0] =", Y[i, 0], file=sys.stderr)

    # B = M.dot(Y)
    # print("B[0, 0] =", B[0, 0], file=sys.stderr)

    # perform multiplication until convergence
    i = 0
    prev_Y = None
    while not _eq_sparse(prev_Y, Y) and i < MAX_I:
        prev_Y = Y.copy()
        Y = Y.tocsc()
        Y = M.dot(Y)
        Y = Y.tolil()
        _sign_normalize(Y, terms2idx, a_pos, a_neg, a_neut)
        i += 1
    ret = _mtx2tlist(Y, terms2idx)
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    return ret
