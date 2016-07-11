#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Awadallah and Radev's method (2010).

USAGE:
esuli_sebstiani.py [OPTIONS] [INPUT_FILES]

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import ANTIRELS, SYNRELS, POSITIVE, NEGATIVE, NEUTRAL, \
    SYNONYM

from bisect import bisect_left
from collections import defaultdict
from Queue import Queue
from threading import Thread

import numpy as np
import sys


##################################################################
# Constants
np.random.seed()
CNT_IDX = 0
PROB_IDX = 1
THRSHLD = 0.5
MAX_STEPS = 10
N_WALKERS = 17
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
        # compute probabilities of edges and create sampling scales
        self._sample_pos2node = defaultdict(dict)
        self._finalize_nodes(a_teleport)

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
        for ilexid in self.germanet.synid2lexids[a_synid]:
            for ilex in self.germanet.lexid2lex[ilexid]:
                # each edge has the form:
                # src_node => trg_node => cnt
                # (we don't differentiate between edge types)
                isrc_node = (ilex, a_pos)
                for itrg_node in trg_nodes:
                    if isrc_node != itrg_node:
                        self.nodes[isrc_node][itrg_node] += 1

    def add_seeds(self, a_terms, a_pol, a_pos):
        """Add polar terms.

        @param a_terms - list of polar terms
        @param a_terms - list of polar terms
        @param a_terms - list of polar terms

        @return

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
        iqueue = Queue(N_WALKERS)
        # start each walker in a separate thread
        for inode in self.nodes.iterkeys():
            # send walkers
            for i in xrange(N_WALKERS):
                iwalker = Thread(None, self._rndm_walk_helper,
                                 str(i), (a_rndm_gen, inode,
                                          iqueue, MAX_STEPS))
                iwalker.start()
                walkers.append(iwalker)
            # join walkers
            for iwalker in walkers:
                iwalker.join()
                assert not iwalker.isAlive(), \
                    "Thread #{:s} exited abnormally.".format(iwalker.name)
            # compute the polarity score of the term
            iscore = 0.
            while not iqueue.empty():
                iscore += iqueue.get()
            # obtain the mean of all scores
            # print("iscore =", repr(iscore), file=sys.stderr)
            iscore /= N_WALKERS
            if iscore > THRSHLD:
                ret[inode] = (iscore, POSITIVE)
            elif -iscore > THRSHLD:
                ret[inode] = (iscore, NEGATIVE)
            else:
                ret[inode] = (iscore, NEUTRAL)
        return ret

    def _rndm_walk_helper(self, a_rndm_gen, a_node,
                          a_queue, a_max_steps):
        """Perform single random walk from the given nodes to the first seed.

        @param a_rndm_gen - generator of random numbers
        @param a_node - node to start the random walk from
        @param a_queue - result queue to be updated
        @param a_max_steps - maximum number of steps to be made

        @return \c void (updates a_queue in place)

        """
        sample = 0.
        inode = a_node
        ipos = ret = 0
        # print("_rndm_walk_helper: a_node", repr(a_node), file=sys.stderr)
        # print("_rndm_walk_helper: a_queue", repr(a_queue), file=sys.stderr)
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

            if ipos >= len(self._sample_pos2node[a_node]):
                break
            a_node = self._sample_pos2node[a_node][ipos]
            # print("_rndm_walk_helper: trg_node =",
            #       repr(a_node), file=sys.stderr)
            if a_node == TELEPORT:
                raise NotImplementedError
            elif a_node in self._seeds:
                ret = self._seeds[a_node]
                break
        a_queue.put(ret)
        # print("_rndm_walk_helper: modified a_queue", repr(a_queue),
        #       file=sys.stderr)

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
    """Extend sentiment lexicons using the  method of Hu and Liu (2004).

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no
      restriction)
    @param a_ext_syn_rels - use extended set of synonymous relations
    @param a_teleport - probability of a random teleport transition

    @return \c void

    """
    sgraph = Graph(a_germanet, a_ext_syn_rels, a_teleport)
    if a_seed_pos == "none":
        a_seed_pos = None
    # initialize seed sets
    sgraph.add_seeds(a_pos, POSITIVE, a_seed_pos)
    sgraph.add_seeds(a_neg, NEGATIVE, a_seed_pos)
    sgraph.add_seeds(a_neut, NEUTRAL, a_seed_pos)
    # perform random walk
    ret = []
    pterms = sgraph.rndm_walk(np.random.uniform)
    for ((iterm, _), (iscore, ipol)) in pterms.iteritems():
        if ipol != NEUTRAL:
            ret.append((iterm, ipol, iscore))
    # convert obtained lex ids back to lexemes
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    return ret
