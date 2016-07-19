#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Rao and Ravichandran's method (2009).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import POSITIVE, NEGATIVE
from graph import Graph

import sys

##################################################################
# Constants


##################################################################
# Methods
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


def rao_label_propagation(a_germanet, a_pos, a_neg, a_neut, a_seed_pos,
                          a_ext_syn_rels, a_teleport):
    """Extend sentiment lexicons using the lbl-prop method of Rao (2009).

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no
      restriction)
    @param a_ext_syn_rels - use extended set of synonymous relations
    @param a_teleport - probability of a random teleport transition

    @return list of polar terms, their polarities, and scores

    """
    pass
