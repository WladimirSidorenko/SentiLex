#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Rao and Ravichandran's method (2009).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from blair_goldensohn import seeds2seedpos
from common import NEGATIVE, POSITIVE, SYNRELS
from rao import POS_IDX, NEG_IDX, NEUT_IDX, POL_IDX, SCORE_IDX

import numpy as np
import sys

##################################################################
# Constants


##################################################################
# Methods
def seeds2synids(a_germanet, a_terms, a_pos):
    """Convert list of lexical terms to synset id's.

    @param a_germanet - GermaNet instance
    @param a_terms - set of terms to check
    @param a_pos - part-of-speech tag of the lexical term or None

    @return list of synset id's

    """
    ret = set()
    for ilex in a_terms:
        for ilexid in a_germanet.lex2lexid[ilex]:
            for isynid in a_germanet.lexid2synids[ilexid]:
                if a_pos is None or a_germanet.synid2pos[isynid] in a_pos:
                    ret.add(isynid)
    return ret


def seedpos_chck(a_germanet, a_term, a_pos):
    """Check whether given term and pos pair is present in Ontology.

    @param a_germanet - GermaNet instance
    @param a_term - lexical term to check
    @param a_pos - part-of-speech tag of the lexical term

    @return \c True if the given term is present in Ontology

    """
    ipos = None
    for lexid in a_germanet.lex2lexid.get(a_term, []):
        for isynid in a_germanet.lexid2synids[lexid]:
            ipos = a_germanet.synid2pos[isynid]
            if ipos == a_pos:
                return True
    return False


def _add_syn(a_scores, a_idx, a_synid, a_germanet, a_term2idx):
    """Update counter of synonyms from a synset.

    @param a_scores - target matrix to update
    @param a_idx - index of the polarity class
    @param a_synid - synset id whose lex counters should be updated
    @param a_germanet - GermaNet instance
    @param a_term2idx - mapping from (term, pos) to matrix index

    @return void

    @note modifies a_scores in place

    """
    ipos = a_germanet.synid2pos[a_synid]
    for ilexid in a_germanet.synid2lexids[a_synid]:
        for ilex in a_germanet.lexid2lex[ilexid]:
            iterm = (ilex, ipos)
            idx = a_term2idx[iterm]
            a_scores[a_idx, idx] += 1


def _add_lex(a_scores, a_idx, a_lexid, a_germanet, a_term2idx):
    """Update counter of synonyms from lexemes.

    @param a_scores - target matrix to update
    @param a_idx - index of the polarity class
    @param a_lexid - id of the lexeme whose counters should be updated
    @param a_germanet - GermaNet instance
    @param a_term2idx - mapping from (term, pos) to matrix index

    @return void

    @note modifies a_scores in place

    """
    ipos = ""
    for isynid in a_germanet.lexid2synid[a_lexid]:
        ipos = a_germanet.synid2pos[isynid]
        for ilex in a_germanet.lexid2lex[a_lexid]:
            iterm = (ilex, ipos)
            idx = a_term2idx[iterm]
            a_scores[a_idx, idx] += 1


def _compute_numerator(a_scores, a_idx, a_seed_synids, a_seed_lexidpos,
                       a_germanet, a_term2idx, a_ext_rel):
    """Compute the number of synsets containing a seed term.

    @param a_scores - target matrix to update
    @param a_idx - index of the polarity class
    @param a_seed_synids - synset id's of the seeds
    @param a_seed_lexidpos - synset id's of the seeds
    @param a_germanet - GermaNet instance
    @param a_term2idx - mapping from (term, pos) to matrix index
    @param a_ext_rel - use an extended set of synonymous relations

    @return \c void

    @note modifies a_scores in place

    """
    idx = 0
    ipos = iterm = None
    # add synonymous relations from synsets
    for isynid in a_seed_synids:
        _add_syn(a_scores, a_idx, isynid, a_germanet, a_term2idx)
        if a_ext_rel:
            for trg_synid, irelname in a_germanet.con_relations[isynid]:
                if irelname in SYNRELS:
                    _add_syn(a_scores, a_idx, trg_synid,
                             a_germanet, a_term2idx)
    # add lexical synonymous relations
    if a_ext_rel:
        src_lexid = set(ilexid
                        for ilex, _ in a_seed_lexidpos
                        for ilexid in a_germanet.lex2lexid)
        for ilexid in src_lexid:
            for itrg, irel in a_germanet.lex_relations[ilexid]:
                if irel in SYNRELS:
                    _add_lex(a_scores, a_idx, itrg,
                             a_germanet, a_term2idx)


def kim_hovy(a_germanet, a_pos, a_neg, a_neut, a_seed_pos,
             a_ext_syn_rels):
    """Extend sentiment lexicons using the min-cut method of Kim/Hovy (2006).

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no
      restriction)
    @param a_ext_syn_rels - use extended set of synonymous relations

    @return list of polar terms, their polarities, and scores

    """
    # construct a graph from GermaNet
    nodes = set((ilex, ipos)
                for isynid, ipos in a_germanet.synid2pos.iteritems()
                for ilexid in a_germanet.synid2lexids[isynid]
                for ilex in a_germanet.lexid2lex[ilexid]
                )
    # estimate prior polarities of the classes
    Z = float(len(nodes))
    if a_seed_pos == "none":
        a_seed_pos = ["adj", "nomen", "verben"]
    else:
        a_seed_pos = [a_seed_pos]

    pos_seedpos = filter(lambda x: seedpos_chck(a_germanet, *x),
                         seeds2seedpos(a_pos, a_seed_pos))
    N_pos = float(len(pos_seedpos))
    P_pos = N_pos / Z
    assert 0. <= P_pos <= 1., \
        "Invalid value of prior positive probability: {:f}".format(P_pos)
    pos_synids = seeds2synids(a_germanet, a_pos, a_seed_pos)

    neg_seedpos = filter(lambda x: seedpos_chck(a_germanet, *x),
                         seeds2seedpos(a_neg, a_seed_pos))
    N_neg = float(len(neg_seedpos))
    P_neg = N_neg / Z
    assert 0. <= P_neg <= 1., \
        "Invalid value of prior negative probability: {:f}".format(P_neg)
    neg_synids = seeds2synids(a_germanet, a_neg, a_seed_pos)

    neut_seedpos = filter(lambda x: seedpos_chck(a_germanet, *x),
                          seeds2seedpos(a_neut, a_seed_pos))
    N_neut = float(len(neut_seedpos))
    P_neut = N_neut / Z
    assert 0. <= P_neut <= 1., \
        "Invalid value of prior neutral probability: {:f}".format(P_neut)
    neut_synids = seeds2synids(a_germanet, a_neut, a_seed_pos)

    # estimate conditional probabilities of terms given classes
    term2idx = {iterm: i for i, iterm in enumerate(nodes)}
    scores = np.zeros((3, len(term2idx)))
    # compute the number of synsets intersecting with the seeds
    # normalize probability scores
    for iseed_synids, iseed_lexidpos, idx \
        in zip((pos_synids, neg_synids, neut_synids),
               (pos_seedpos, neg_seedpos, neut_seedpos),
               (POS_IDX, NEG_IDX, NEUT_IDX)):
        _compute_numerator(scores, idx, iseed_synids, iseed_lexidpos,
                           a_germanet, term2idx, a_ext_syn_rels)
    # normalize probabilities
    scores[POS_IDX, :] *= P_pos / (float(len(pos_synids)) or 1.)
    scores[NEG_IDX, :] *= P_neg / (float(len(neg_synids)) or 1.)
    scores[NEUT_IDX, :] *= P_neut / (float(len(neut_synids)) or 1.)
    # check
    assert np.max(np.max(scores, axis=1), axis=0) <= 1., \
        "Ivalid maximum probability value."
    assert np.min(np.min(scores, axis=1), axis=0) >= 0., \
        "Ivalid minimum probability value: {:f}.".format(
            np.min(np.min(scores, axis=1), axis=0))
    # determine classes
    idx = 0
    ret = [(w, POSITIVE, FMAX) for w in a_pos] \
        + [(w, NEGATIVE, FMIN) for w in a_neg]
    lex2lidx = {el[0]: i for i, el in enumerate(ret)}
    classes = np.argmax(scores, axis=0)
    for (ilex, ipos), idx in term2idx.iteritems():
        if ilex in a_pos or ilex in a_neg:
            continue
        if classes[idx] == POS_IDX:
            ipol = POSITIVE
            iscore = scores[POS_IDX, idx]
        elif classes[idx] == NEG_IDX:
            ipol = NEGATIVE
            iscore = scores[NEG_IDX, idx]
        else:
            continue
        if iscore == 0.:
            continue
        if ilex in lex2lidx:
            idx = lex2lidx[ilex]
            if abs(iscore) > abs(ret[idx][SCORE_IDX]):
                ret[idx][POL_IDX] = ipol
                ret[idx][SCORE_IDX] = iscore
        else:
            lex2lidx[ilex] = len(ret)
            ret.append([ilex, ipol, iscore])
    return ret
