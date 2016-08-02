#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating a Graph from GermaNet.

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import SYNRELS, POSITIVE, NEGATIVE, NEUTRAL

from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import numpy as np
import sys

##################################################################
# Constants
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

    def add_seeds(self, a_terms, a_pol, a_pos):
        """Add polar terms.

        @param a_terms - list of polar terms
        @param a_pol - polarity of the polar terms
        @param a_pos - required part-of-speech class of the polar terms

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
        for iterm in self._term2termpos(a_terms, a_pos):
            self._seeds[iterm] = a_pol

    def bfs(self, a_start, a_end, a_parents):
        """Preform breadth-first search.

        @param a_start - node to start the BFS from
        @param a_end - node to end the BFS with
        @param a_parents - mapping from nodes to their parents to be populated

        @return \c True if ``a_end`` was reached during the search

        """
        return self._bfs(self.nodes, a_start, a_end, a_parents)

    def rndm_walk(self, a_rndm_gen):
        """Perform random walk from all nodes to the seeds.

        @param a_rndm_gen - generator of random numbers

        @return dict - mapping from terms to their scores and polarity values

        """
        ret = {}
        iscore = 0.
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
            # assert inode != ("gut", "adj") or iscore == 1.
            if iscore > THRSHLD:
                ret[inode] = (iscore, POSITIVE)
            elif -iscore > THRSHLD:
                ret[inode] = (iscore, NEGATIVE)
            else:
                ret[inode] = (iscore, NEUTRAL)
        print("Processing nodes... done", file=sys.stderr)
        return ret

    def min_cut(self, a_nodes1, a_nodes2, a_pos=None):
        """Find minumum cut that separates ``a_nodes1`` form ``a_nodes2``.

        @param a_nodes1 - first set of nodes
        @param a_nodes2 - second set of nodes
        @param a_pos - part-of-speech of the nodes

        @return (set, set) - min-cut partitioning of the graph separating
          ``a_nodes1`` from ``a_nodes2``

        """
        # convert terms to (terms, part-of-speech) pairs
        if not np.all([isinstance(inode, tuple) for inode in a_nodes1]):
            assert a_nodes1
            a_nodes1 = frozenset([iterm
                                  for iterm
                                  in self._term2termpos(a_nodes1, a_pos)])
            assert np.all([isinstance(inode, tuple) for inode in a_nodes1])
        # print("a_nodes1 =", repr(a_nodes1), file=sys.stderr)
        if not np.all([isinstance(inode, tuple) for inode in a_nodes2]):
            a_nodes2 = frozenset([iterm
                                  for iterm in self._term2termpos(a_nodes2,
                                                                  a_pos)])
            assert np.all([isinstance(inode, tuple) for inode in a_nodes2])
        # print("a_nodes2 =", repr(a_nodes2), file=sys.stderr)

        hypnode1 = orig_edges1 = hypnode2 = orig_edges2 = None
        try:
            # replace seed terms nodes with hyper-nodes
            hypnode1, orig_edges1 = self._nodes2hypernode(a_nodes1)
            hypnode2, orig_edges2 = self._nodes2hypernode(a_nodes2)
            # min-cut the graph
            mcs, cut_edges, ret1, ret2 = self._min_cut(a_nodes1, a_nodes2)
            assert a_nodes1 in ret1, \
                "Source node not found in the first partition."
            # print("a_nodes2 =", repr(a_nodes2), file=sys.stderr)
            # print("ret2 =", repr(ret2), file=sys.stderr)
            assert a_nodes2 in ret2, \
                "Sink node not found in the second partition."
            # replace artificial hyper-nodes with their original equivalents
            ret1.remove(a_nodes1)
            ret1.update(a_nodes1)
            ret2.remove(a_nodes2)
            ret2.update(a_nodes2)
        finally:
            # replace hyper-nodes with the original vertices
            self._hypernode2nodes(hypnode1, orig_edges1)
            self._hypernode2nodes(hypnode2, orig_edges2)
        return (mcs, cut_edges, ret1, ret2)

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
                    if rel_type in SYNRELS:
                        trg_pos = ""
                        for isyn_id in self.germanet.lexid2synids[trg_lexid]:
                            trg_pos = self.germanet.synid2pos[isyn_id]
                            break
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

    def _bfs(self, a_graph, a_start, a_end, a_parents):
        """Preform breadth-first search on the given graph.

        @param a_graph - graph to perform the search on
        @param a_start - node to start the BFS from
        @param a_end - node to end the BFS with
        @param a_parents - mapping from nodes to their parents to be populated

        @return \c True if ``a_end`` was reached during the search

        """
        inode = None
        iqueue = [a_start]
        visited = set(iqueue)
        a_parents[a_start] = None
        while iqueue:
            inode = iqueue.pop(0)
            for itrg, iwght in a_graph[inode].iteritems():
                if not itrg in visited and iwght > 0:
                    a_parents[itrg] = inode
                    iqueue.append(itrg)
                    visited.add(itrg)
        return a_end in visited

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
                i = len(self._nsamples[src_node])
                self._nsamples[src_node].append(1.)
                self._sample_pos2node[src_node][i] = TELEPORT

    def _hypernode2nodes(self, a_hypnode, a_orig_edges):
        """Split a hyper-node into atomic nodes.

        @param a_hypnode - hyper-node to split
        @param a_orig_edges - original graph edges

        @return \c void

        """
        if a_hypnode is None:
            return
        self.nodes.pop(a_hypnode)
        for isrc, iedges in a_orig_edges.iteritems():
            for itrg, iwght in iedges.iteritems():
                self.nodes[isrc][itrg] = iwght
            if a_hypnode in self.nodes[isrc]:
                self.nodes[isrc].pop(a_hypnode)

    def _min_cut(self, a_s, a_t):
        """Find minumum cut that separates ``a_nodes1`` from ``a_nodes2``.

        @param a_s - s node
        @param a_t - t node

        @return (int, set, set) - min-cut score and partitioning of the graph
          separating ``a_nodes1`` from ``a_nodes2``

        """
        # reverse the graph
        rGraph = defaultdict(lambda:
                             defaultdict(int))
        for isrc, iedges in self.nodes.iteritems():
            for itrg, iwght in iedges.iteritems():
                rGraph[itrg][isrc] = iwght
        # augment the flow while there is a path from source to sink
        parents = {}
        u = v = None
        min_flow = inf = float("inf")
        while self._bfs(rGraph, a_s, a_t, parents):
            v = a_t
            min_flow = inf
            # obtain minimum flow
            while not v is a_s:
                u = parents[v]
                min_flow = min(rGraph[u][v], min_flow)
                v = u
            # update residual capacities of the edges and reverse the edges
            # along the path
            assert min_flow < inf, "Minimum flow is infinite."
            v = a_t
            while not v is a_s:
                u = parents[v]
                rGraph[u][v] -= min_flow
                rGraph[v][u] += min_flow
                v = u
            parents.clear()
        # find vertices reachable from source
        min_cut_score = 0
        cut_edges = set()
        for isrc in parents:
            for itrg, iwght in self.nodes[isrc].iteritems():
                if not itrg in parents and iwght > 0:
                    cut_edges.add((isrc, itrg))
                    cut_edges.add((itrg, isrc))
                    min_cut_score += iwght
        # split the graph into two partitions, searching for the nodes
        # available from source and sink
        parents.clear()
        # remove edges belonging to the min cut (i.e., cut the graph)
        for isrc, itrg in cut_edges:
            self.nodes[isrc].pop(itrg, None)
        self.bfs(a_s, a_t, parents)
        nodes1 = set(parents.iterkeys())
        parents.clear()
        self.bfs(a_t, a_s, parents)
        nodes2 = set(inode for inode in parents if inode not in nodes1)
        return (min_cut_score, cut_edges, nodes1, nodes2)

    def _nodes2hypernode(self, a_nodes):
        """Create a hyper-node from all nodes pertaining to a set.

        @param a_nodes - set of nodes to make a hyper-node from

        @return 2-tuple: id of hyper-node and dict of replaced edges

        """
        if not isinstance(a_nodes, frozenset):
            a_nodes = frozenset(a_nodes)
        orig_edges = defaultdict(lambda: defaultdict(int))
        # relink all edges outgoing from ``a_nodes`` to the new hyper-node
        for inode in a_nodes:
            for itrg, iwght in self.nodes[inode].iteritems():
                self.nodes[a_nodes][itrg] += iwght
            orig_edges[inode] = self.nodes[inode]
            self.nodes.pop(inode)
        # relink all edges incident to any of the ``a_nodes`` to the new
        # hyper-node
        for inode in self.nodes:
            # first, remember the original links
            for itrg, iwght in self.nodes[inode].iteritems():
                if itrg in a_nodes:
                    orig_edges[inode][itrg] = iwght
            # then, modify the graph
            for itrg in orig_edges[inode]:
                self.nodes[inode][a_nodes] = orig_edges[inode][itrg]
                self.nodes[inode].pop(itrg)
        return (a_nodes, orig_edges)

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
        ipos = ret = 0
        if a_node in self._seeds:
            ret = self._seeds[a_node]
            a_max_steps = 0
        for i in xrange(a_max_steps):
            # make a sample
            sample = a_rndm_gen()
            ipos = bisect_left(self._nsamples[a_node], sample)
            # break out if no transitions are left
            if ipos >= len(self._sample_pos2node[a_node]):
                break
            a_node = self._sample_pos2node[a_node][ipos]
            if a_node == TELEPORT:
                # teleport to a random node
                ipos = np.random.choice(self._n_nodes)
                a_node = self._node_keys[ipos]
            if a_node in self._seeds:
                ret = self._seeds[a_node]
                break
        a_scores[a_i] = ret

    def _term2termpos(self, a_terms, a_pos):
        """Add polar terms.

        @param a_terms - list of terms
        @param a_pos - required part-of-speech class of the terms

        @return iterator over terms and their parts of speech

        """
        ipos = None
        for iterm in a_terms:
            if isinstance(iterm, tuple):
                yield iterm
            else:
                for ilexid in self.germanet.lex2lexid[iterm]:
                    for isynid in self.germanet.lexid2synids[ilexid]:
                        ipos = self.germanet.synid2pos[isynid]
                        if a_pos is None or a_pos == ipos:
                            yield (iterm, ipos)
