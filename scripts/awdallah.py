#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Awadallah and Radev's method (2010).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import POSITIVE, NEGATIVE, NEUTRAL
from graph import Graph

import numpy as np


##################################################################
# Constants
np.random.seed()


##################################################################
# Methods
def awdallah(a_germanet, a_pos, a_neg, a_neut, a_seed_pos,
             a_ext_syn_rels, a_teleport):
    """Extend sentiment lexicons using the  method of Awdallah (2010).

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
    sgraph = Graph(a_germanet, a_ext_syn_rels, a_teleport)
    if a_seed_pos == "none":
        a_seed_pos = None
    # initialize seed sets
    sgraph.add_seeds(a_pos, POSITIVE, a_seed_pos)
    sgraph.add_seeds(a_neg, NEGATIVE, a_seed_pos)
    sgraph.add_seeds(a_neut, NEUTRAL, a_seed_pos)
    # if "gut" in a_pos and a_seed_pos:
    #     assert sgraph._seeds[("gut", "adj")] == 1.
    # if "schlecht" in a_neg and a_seed_pos:
    #     assert sgraph._seeds[("schlecht", "adj")] == -1.
    # perform random walk
    ret = []
    pterms = sgraph.rndm_walk(np.random.uniform)
    for ((iterm, _), (iscore, ipol)) in pterms.iteritems():
        if ipol != NEUTRAL:
            ret.append((iterm, ipol, iscore))
    # sort obtained lexemes by their scores
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    return ret
