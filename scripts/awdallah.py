#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Awadallah and Radev's method (2010).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import ANTIRELS, SYNRELS, POSITIVE, NEGATIVE, NEUTRAL, \
    SYNONYM

from bisect import bisect_left
from collections import defaultdict
from itertools import chain

import numpy as np
import sys


##################################################################
# Constants
np.random.seed()
CNT_IDX = 0
PROB_IDX = 1
THRSHLD = 0.5
MAX_STEPS = 17
N_WALKERS = 7
TELEPORT = "***TELEPORT***"


##################################################################
# Classes
class Graph(object):
    """Synset graph.

    """

    def __init__(self, a_germanet, a_ext_rel=False, a_teleport=False):
        """Construct synset graph.

        @param a_germanet - GermaNet instance to construct the graph from
        @param a_ext_rel - use extended set of synonymous relations
        @param a_teleport - probability of a random teleport transition
        """
        self._seeds = {}
        self.germanet = a_germanet
        # nodes have the structure:
        # src_node => trg_node => cnt
        self.nodes = defaultdict(lambda:
                                 defaultdict(int))
        self._nsamples = defaultdict(list)
        for isynid, ipos in self.germanet.synid2pos.iteritems():
            self._add_edges(isynid, ipos, a_ext_rel)
        assert self.nodes[("beachtlich", "adj")][("erklecklich", "adj")] == 1.
        assert self.nodes[("hinsetzen",
                           "verben")][("niedersetzen", "verben")] == 2.
        assert self.nodes[("zweiteilung",
                           "nomen")][("dichotomie", "nomen")] == 1.
        if a_ext_rel:
            assert self.nodes[("hünenhaft",
                               "adj")][("hüne", "nomen")] == 1.
            assert ("sonderbar", "adj") \
                not in self.nodes[("kafkaesk", "adj")]
            assert self.nodes[("sonderbar",
                               "adj")][("kafkaesk", "adj")] == 1.

        self._node_keys = self.nodes.keys()
        self._n_nodes = len(self.nodes)
        # compute probabilities of edges and create sampling scales
        self._sample_pos2node = defaultdict(dict)
        self._finalize_nodes(a_teleport)
        idx = 0
        for inode in self.nodes.iterkeys():
            assert not self._nsamples[inode] or \
                np.isclose(self._nsamples[inode][-1], [1.])
            idx = len(self._nsamples[inode]) - 1
            assert not a_teleport \
                or self._sample_pos2node[inode][idx] == TELEPORT

    def _add_edges(self, a_synid, a_pos, a_ext_rel):
        """Add edges to the node's adjacency matrix.

        @param a_synid - id of source synset
        @param a_pos - part-of-speech class of source synset
        @param a_ext_rel - use extended set of synonymous relations

        """
        isrc_node = None
        trg_nodes = set((ilex, a_pos)
                        for ilexid in self.germanet.synid2lexids[a_synid]
                        for ilex in self.germanet.lexid2lex[ilexid])
        if a_ext_rel:
            trg_pos = None
            for trg_synid, rel_type in self.germanet.con_relations[a_synid]:
                trg_pos = self.germanet.synid2pos[trg_synid]
                if rel_type in SYNRELS:
                    for ilexid in self.germanet.synid2lexids[trg_synid]:
                        for ilex in self.germanet.lexid2lex[ilexid]:
                            trg_nodes.add((ilex, trg_pos))
        # add target nodes to each lexeme pertaining to the given synset
        trg_pos = ""
        ext_syn_nodes = []
        for ilexid in self.germanet.synid2lexids[a_synid]:
            if a_ext_rel:
                for trg_lexid, rel_type in self.germanet.lex_relations[ilexid]:
                    trg_pos = ""
                    for isyn_id in self.germanet.lexid2synids[trg_lexid]:
                        trg_pos = self.germanet.synid2pos[isyn_id]
                        break
                    if rel_type in SYNRELS:
                        for ilex in self.germanet.lexid2lex[trg_lexid]:
                            ext_syn_nodes.append((ilex, trg_pos))
            for ilex in self.germanet.lexid2lex[ilexid]:
                # each edge has the form:
                # src_node => trg_node => cnt
                # (we don't differentiate between edge types)
                isrc_node = (ilex, a_pos)
                for itrg_node in chain(trg_nodes, ext_syn_nodes):
                    if isrc_node != itrg_node:
                        self.nodes[isrc_node][itrg_node] += 1
            if ext_syn_nodes:
                del ext_syn_nodes[:]

    def add_seeds(self, a_terms, a_pol, a_pos):
        """Add polar terms.

        @param a_terms - list of polar terms
        @param a_terms - list of polar terms
        @param a_terms - list of polar terms

        @return \c void

        """
        if a_pol == POSITIVE:
            a_pol = 1
        elif a_pol == NEGATIVE:
            a_pol = -1
        elif a_pol == NEUTRAL:
            a_pol = 0
        else:
            raise ValueError("Unknown polarity value: {:s}".format(a_pol))
        ipos = None
        for iterm in a_terms:
            for ilexid in self.germanet.lex2lexid[iterm]:
                for isynid in self.germanet.lexid2synids[ilexid]:
                    ipos = self.germanet.synid2pos[isynid]
                    if a_pos is None or a_pos == ipos:
                        self._seeds[(iterm, ipos)] = a_pol

    def rndm_walk(self, a_rndm_gen):
        """Perform random walk from all nodes to the seeds.

        @param a_rndm_gen - generator of random numbers

        @return dict - mapping from terms to their scores and polarity values

        """
        ret = {}
        iscore = 0.
        walkers = []
        iwalker = None
        N = len(self.nodes)
        iscores = np.zeros(N_WALKERS)
        # start each walker in a separate thread
        print("Processing nodes... started", file=sys.stderr)
        for i, inode in enumerate(self.nodes.iterkeys()):
            # print("{:d} of {:d}".format(i, N), end="\r", file=sys.stderr)
            # send walkers
            for i in xrange(N_WALKERS):
                self._rndm_walk_helper(a_rndm_gen, inode,
                                       iscores, i, MAX_STEPS)
            # obtain the mean of all scores
            assert inode != ("mau", "adj") or np.any(iscore == -1.
                                                     for iscore in iscores)
            iscore = iscores.mean()
            iscores *= 0.
            assert inode != ("schlecht", "adj") or iscore == -1.
            assert inode != ("gut", "adj") or iscore == 1.
            if iscore > THRSHLD:
                ret[inode] = (iscore, POSITIVE)
            elif -iscore > THRSHLD:
                ret[inode] = (iscore, NEGATIVE)
            else:
                ret[inode] = (iscore, NEUTRAL)
        print("Processing nodes... done", file=sys.stderr)
        return ret

    def _rndm_walk_helper(self, a_rndm_gen, a_node,
                          a_scores, a_i, a_max_steps):
        """Perform single random walk from the given nodes to the first seed.

        @param a_rndm_gen - generator of random numbers
        @param a_node - node to start the random walk from
        @param a_scores - result list of scores to be updated
        @param a_i - index in the score list at which the result should be
          stored
        @param a_max_steps - maximum number of steps to be made

        @return \c void (updates a_queue in place)

        """
        sample = 0.
        inode = a_node
        ipos = ret = 0
        # print("_rndm_walk_helper: a_node", repr(a_node), file=sys.stderr)
        # print("_rndm_walk_helper: a_queue", repr(a_queue), file=sys.stderr)
        if a_node in self._seeds:
            ret = self._seeds[a_node]
            a_max_steps = 0
        for i in xrange(a_max_steps):
            # print("_rndm_walk_helper: self._nsamples[a_node]",
            #       repr(self._nsamples[a_node]), file=sys.stderr)
            # print("_rndm_walk_helper: self._sample_pos2node[a_node]",
            #       repr(self._sample_pos2node[a_node]), file=sys.stderr)
            # make a sample
            sample = a_rndm_gen()
            ipos = bisect_left(self._nsamples[a_node], sample)
            # print("_rndm_walk_helper: sample =",
            #       repr(sample), file=sys.stderr)
            # print("_rndm_walk_helper: ipos =",
            #       repr(ipos), file=sys.stderr)
            # break out if no transitions are left
            if ipos >= len(self._sample_pos2node[a_node]):
                break
            a_node = self._sample_pos2node[a_node][ipos]
            # print("_rndm_walk_helper: trg_node =",
            #       repr(a_node), file=sys.stderr)
            if a_node == TELEPORT:
                # teleport to a random node
                ipos = np.random.choice(self._n_nodes)
                a_node = self._node_keys[ipos]
            if a_node in self._seeds:
                ret = self._seeds[a_node]
                break
        a_scores[a_i] = ret

    def _finalize_nodes(self, a_teleport):
        """Compute probabilities of transitions.

        @param a_teleport - probability of a random teleport transition

        @return \c void

        """
        z = 0.
        reciprocal_teleport = 1. - a_teleport
        for src_node, trg_nodes in self.nodes.iteritems():
            z = float(sum(trg_nodes.itervalues()))
            istart = 0.
            for i, (itrg_node, icnt) in enumerate(trg_nodes.iteritems()):
                istart += reciprocal_teleport * float(icnt) / z
                self._nsamples[src_node].append(istart)
                self._sample_pos2node[src_node][i] = itrg_node
            if a_teleport:
                assert istart < 1., \
                    "No probability mass left for teleport transition."
                self._nsamples[src_node].append(1.)
                self._sample_pos2node[src_node][i] = TELEPORT


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
    assert sgraph._seeds[("gut", "adj")] == 1.
    assert sgraph._seeds[("schlecht", "adj")] == -1.
    # perform random walk
    ret = []
    pterms = sgraph.rndm_walk(np.random.uniform)
    for ((iterm, _), (iscore, ipol)) in pterms.iteritems():
        if ipol != NEUTRAL:
            ret.append((iterm, ipol, iscore))
    # sort obtained lexemes by their scores
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    return ret
