#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Takamura's method (2005).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import lemmatize, POSITIVE, NEGATIVE, TOKENIZER, \
    SYNRELS, ANTIRELS, NEGATORS, STOP_WORDS, FORM2LEMMA, \
    TAB_RE, ENCODING, check_word
from germanet import normalize
from ising import Ising, ITEM_IDX, WGHT_IDX, HAS_FXD_WGHT, FXD_WGHT_IDX

from itertools import chain, combinations
from math import isnan
import codecs
import sys


##################################################################
# Methods
def _tkm_add_germanet(ising, a_germanet):
    """Add lexical nodes from GermaNet to the Ising spin model

    @param a_ising - instance of the Ising spin model
    @param a_germanet - GermaNet instance

    @return \c void

    """
    # add all lemmas from the `FORM2LEMMA` dictionary
    for ilemma in FORM2LEMMA.itervalues():
        if ilemma not in STOP_WORDS:
            ising.add_node(ilemma)
    # add all lemmas from synsets
    for ilexid in a_germanet.lexid2synids.iterkeys():
        for ilex in a_germanet.lexid2lex[ilexid]:
            ising.add_node(ilex)
    # establish links between synset words and lemmas appearing in
    # examples and definitions
    def_lexemes = []
    negation_seen = False
    for isynid, (idef, iexamples) in a_germanet.synid2defexmp.iteritems():
        def_lexemes = [lemmatize(iword, a_prune=False)
                       for itxt in chain(idef, iexamples)
                       for iword in TOKENIZER.tokenize(itxt)]
        def_lexemes = [ilexeme
                       for ilexeme in def_lexemes
                       if ilexeme
                       and ising.item2nid.get(ilexeme, None) is not None]
        if def_lexemes:
            negation_seen = False
            for idef_lex in def_lexemes:
                if idef_lex in NEGATORS:
                    negation_seen = True
                    continue
                elif idef_lex in STOP_WORDS:
                    continue
                for ilexid in a_germanet.synid2lexids[isynid]:
                    for ilex in a_germanet.lexid2lex[ilexid]:
                        ising.add_edge(ilex,
                                       idef_lex, -1. if negation_seen else 1.)
    # establish links between synset lemmas based on the lexical
    # relations
    iwght = 1.
    lemmas1 = lemmas2 = None
    for ifrom, irelset in a_germanet.lex_relations.iteritems():
        lemmas1 = a_germanet.lexid2lex.get(ifrom)
        assert lemmas1 is not None, "No lemma found for id {:s}".format(ifrom)
        for ito, irel in irelset:
            lemmas2 = a_germanet.lexid2lex.get(ito)
            assert lemmas2 is not None, \
                "No lemma found for id {:s}".format(ito)
            if irel in SYNRELS:
                iwght = 1.
            elif irel in ANTIRELS:
                iwght = -1.
            else:
                continue
            for ilemma1 in lemmas1:
                for ilemma2 in lemmas2:
                    ising.add_edge(ilemma1, ilemma2, iwght)
    # establish links between synset lemmas based on the con relations
    for ifrom, irelset in a_germanet.con_relations.iteritems():
        # iterate over all lexemes pertaining to the first synset
        for ilex_id1 in a_germanet.synid2lexids[ifrom]:
            lemmas1 = a_germanet.lexid2lex.get(ilex_id1)
            assert lemmas1 is not None, \
                "No lemma found for id {:s}".format(ifrom)
            # iterate over target synsets and their respective relations
            for ito, irel in irelset:
                if irel in SYNRELS:
                    iwght = 1.
                elif irel in ANTIRELS:
                    iwght = -1.
                else:
                    continue
                # iterate over all lexemes pertaining to the second synset
                for ilex_id2 in a_germanet.synid2lexids[ito]:
                    lemmas2 = a_germanet.lexid2lex.get(ilex_id2)
                    assert lemmas2 is not None, \
                        "No lemma found for id {:s}".format(ito)
                    for ilemma1 in lemmas1:
                        for ilemma2 in lemmas2:
                            ising.add_edge(ilemma1, ilemma2, iwght)
    # establish links between lemmas which pertain to the same synset
    ilexemes = set()
    for ilex_ids in a_germanet.synid2lexids.itervalues():
        ilexemes = set([ilex
                        for ilex_id in ilex_ids
                        for ilex in a_germanet.lexid2lex[ilex_id]])
        # generate all possible (n choose 2) combinations of lexemes
        # and put links between them
        for ilemma1, ilemma2 in combinations(ilexemes, 2):
            ising.add_edge(ilemma1, ilemma2, 1.)


def _tkm_add_corpus(ising, a_cc_file):
    """Add lexical nodes from corpus to the Ising spin model

    @param a_ising - instance of the Ising spin model
    @param a_cc_file - file containing conjoined word pairs extracted from
      corpus

    @return \c void

    """
    ifields = []
    iwght = 1.
    ilemma1 = ilemma2 = ""
    with codecs.open(a_cc_file, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            ifields = TAB_RE.split(iline)
            if len(ifields) != 3:
                continue
            ilemma1, ilemma2, iwght = ifields
            if check_word(ilemma1) and check_word(ilemma2):
                ising.add_edge(normalize(ilemma1),
                               normalize(ilemma2), float(iwght),
                               a_add_missing=True)


def takamura(a_germanet, a_N, a_cc_file, a_pos, a_neg, a_neut, a_plot=None):
    """Method for generating sentiment lexicons using Takamura's approach.

    @param a_germanet - GermaNet instance
    @param a_N - number of terms to extract
    @param a_cc_file - file containing coordinatively conjoined phrases
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded
    @param a_plot - name of file in which generated statics plots should be
                    saved (None if no plot should be generated)

    @return \c 0 on success, non-\c 0 otherwise

    """
    # estimate the number of terms to extract
    seed_set = a_pos | a_neg
    # create initial empty network
    ising = Ising()
    # populate network from GermaNet
    print("Adding GermaNet synsets...", end="", file=sys.stderr)
    _tkm_add_germanet(ising, a_germanet)
    print("done (Ising model has {:d} nodes)".format(ising.n_nodes),
          file=sys.stderr)
    # populate network from corpus
    print("Adding coordinate phrases from corpus...",
          end="", file=sys.stderr)
    _tkm_add_corpus(ising, a_cc_file)
    print("done (Ising model has {:d} nodes)".format(ising.n_nodes),
          file=sys.stderr)
    # reweight edges
    ising.reweight()
    # set fixed weights for words pertaining to the positive, negative, and
    # neutral set
    for ipos in a_pos:
        if ipos in ising:
            ising[ipos][FXD_WGHT_IDX] = 1.
        else:
            ising.add_node(ipos, 1.)
        ising[ipos][HAS_FXD_WGHT] = 1
    for ineg in a_neg:
        if ineg in ising:
            ising[ineg][FXD_WGHT_IDX] = -1.
        else:
            ising.add_node(ineg, -1.)
        ising[ineg][HAS_FXD_WGHT] = 1
    for ineut in a_neut:
        if ineut in ising:
            ising[ineut][FXD_WGHT_IDX] = 0.
        else:
            ising.add_node(ineut, 0.)
        ising[ineut][HAS_FXD_WGHT] = 1
    ising.train(a_plot=a_plot)
    # nodes = [inode[ITEM_IDX]
    # for inode in sorted(ising.nodes, key = lambda x: x[WGHT_IDX])
    #              if inode[ITEM_IDX] not in seed_set]
    seed_set |= a_neut
    nodes = [inode
             for inode in sorted(ising.nodes,
                                 key=lambda x: abs(x[WGHT_IDX]), reverse=True)
             if inode[ITEM_IDX] not in seed_set]
    seed_set.clear()
    # populate polarity sets and flush all terms to an external file
    i = 0
    if a_N < 0:
        a_N = len(nodes)

    # generate final set of polar terms
    max_w = max(inode[WGHT_IDX] for inode in nodes) + 1.
    min_w = max(inode[WGHT_IDX] for inode in nodes) - 1.
    # add all original seed terms
    ret = [(iterm, POSITIVE, max_w) for iterm in a_pos] + \
          [(iterm, NEGATIVE, min_w) for iterm in a_neg]
    # add remaining automatically derived terms
    for inode in nodes:
        if isnan(inode[WGHT_IDX]):
            print(inode[ITEM_IDX].encode(ENCODING), "\t", inode[WGHT_IDX],
                  file=sys.stderr)
        else:
            if i < a_N:
                if inode[WGHT_IDX] > 0:
                    ret.append((inode[ITEM_IDX], POSITIVE, inode[WGHT_IDX]))
                elif inode[WGHT_IDX] < 0:
                    ret.append((inode[ITEM_IDX], NEGATIVE, inode[WGHT_IDX]))
                else:
                    continue
                i += 1
            else:
                break
    return ret
