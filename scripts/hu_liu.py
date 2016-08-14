#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating sentiment lexicon using Hu and Liu's method (2004).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import ANTIRELS, SYNRELS, POSITIVE, NEGATIVE, NEUTRAL, \
    ANTONYM, POL2OPPOSITE

import logging
import sys

##################################################################
# Constants
LOGGER = logging
# The below does not work with Python version older than 2.7.7
# LOGGER = logging.getLogger("Hu-Liu")
# LOGGER.setLevel(logging.DEBUG)
# LOGGER.addHandler(logging.StreamHandler())


##################################################################
# Class
class PolarTerm(object):
    """Polar term struct for storing in sets.

    """

    def __init__(self, a_term_id, a_pol, a_wght=1.):
        """Class constructor.

        @param a_term_id - GermaNet id of the lexical term
        @param a_pol - term's polarity class
        @param a_wght - term's weight

        """
        self.term_id = a_term_id
        self.pol = a_pol
        self.wght = a_wght

    def __eq__(self, a_pterm):
        """Comparison operator.

        @param a_pterm - another polar term

        @return True if both terms are equal, False otherwise

        """
        return self.term_id == a_pterm.term_id

    def __hash__(self):
        """Hash function.

        @return hash value of the polar term

        """
        return hash(self.term_id)

    def __repr__(self):
        """Return representation of the given object.

        @return string representation of the given object

        """
        return '<' + self.__class__.__name__ + \
            "term_id = {:s} ".format(self.term_id) + \
            "polarity = {:s} ".format(self.pol) + \
            "weight = {:f} ".format(self.wght) + '>'


##################################################################
# Methods
def _lexemes2polterms(a_germanet, a_lexemes, a_pol, a_pos=None):
    """Convert lexemes to a set of polar terms.

    @param a_germanet - GermaNet instance
    @param a_lexemes - set of lexemes
    @param a_pol - polarity class of lexemes
    @param a_pos - part-od-speech of lexemes to consider
      (None for no restriction)

    @return set of ``PolarTerm`` items

    """
    return set(PolarTerm(ilex_id, a_pol)
               for ilex in a_lexemes
               for ilex_id in a_germanet.lex2lexid.get(ilex, set())
               for isyn_id in a_germanet.lexid2synids.get(ilex_id, [])
               if a_pos is None or a_pos == a_germanet.synid2pos[isyn_id])


def _add_lexemes(a_germanet, a_synid, a_pol, a_wght,
                 a_new_terms, a_pol_terms):
    """Add lexemes pertaining to the given synset.

    @param a_germanet - GermaNet instance
    @param a_synid - id of the synonym being analyzed
    @param a_pol - polarity of the lexical term being analyzed
    @param a_wght - weight for the new terms
    @param a_new_terms - set of newly added terms (to be enriched)
    @param a_pol_terms - target set of polar terms (to be enriched)

    @return \c void

    """
    same_pos = True
    ipolterm = trg_synid = trg_pos = None
    src_pos = a_germanet.synid2pos[a_synid]
    for src_lexid in a_germanet.synid2lexids[a_synid]:
        same_pos = True
        # need this mapping, since the same word might have multiple lex ids
        for src_lex in a_germanet.lexid2lex[src_lexid]:
            for trg_lexid in a_germanet.lex2lexid[src_lex]:
                for trg_synid in a_germanet.lexid2synids[trg_lexid]:
                    if a_germanet.synid2pos[trg_synid] != src_pos:
                        same_pos = False
                        break
                if not same_pos:
                    break
                ipolterm = PolarTerm(trg_lexid, a_pol, a_wght)
                if ipolterm not in a_pol_terms:
                    a_new_terms.add(ipolterm)
                    a_pol_terms.add(ipolterm)


def _analyze_term(a_germanet, a_lexid, a_pol, a_wght,
                  a_new_terms, a_pol_terms, a_expanded_syn_rels):
    """Add synonyms and antonyms of the given term to the pol sets.

    @param a_germanet - GermaNet instance
    @param a_lexid - id of the lexical term being analyzed
    @param a_pol - polarity of the lexical term being analyzed
    @param a_wght - weight for the new terms
    @param a_new_terms - set of newly added terms (to be enriched)
    @param a_pol_terms - target set of polar terms (to be enriched)
    @param a_expanded_syn_rels - use expanded set of synonymous
      relations

    @return \c void

    """
    LOGGER.debug("analyzing term {:s} "
                 "with polarity {:s}".format(
                     a_lexid, a_pol))
    # add synonyms to the set with the same polarity
    # print("a_lexid =", repr(a_lexid), file=sys.stderr)
    ipolterm = None
    for src_synid in a_germanet.lexid2synids[a_lexid]:
        # print("src_synid =", repr(src_synid), file=sys.stderr)
        LOGGER.debug("considering synset: {:s}".format(
            repr(src_synid)))
        _add_lexemes(a_germanet, src_synid, a_pol, a_wght,
                     a_new_terms, a_pol_terms)
        if a_expanded_syn_rels:
            for trg_synid, rel_type in a_germanet.con_relations[src_synid]:
                if rel_type in SYNRELS:
                    for ilex_id in a_germanet.synid2lexids[trg_synid]:
                        ipolterm = PolarTerm(ilex_id, a_pol, a_wght)
                        if ipolterm not in a_pol_terms:
                            a_new_terms.add(ipolterm)
                            a_pol_terms.add(ipolterm)
    # add antonyms to the set with the opposite polarity
    if a_pol == NEUTRAL and not a_expanded_syn_rels:
        return
    ipolterm = None
    LOGGER.debug("*** lex_relations = {:s}".format(
        repr(a_germanet.lex_relations[a_lexid])))
    for ito, ireltype in a_germanet.lex_relations[a_lexid]:
        if a_pol != NEUTRAL and ireltype in ANTIRELS:
            ipolterm = PolarTerm(ito, POL2OPPOSITE[a_pol], a_wght)
        # elif a_expanded_syn_rels and ireltype in SYNRELS:
        #     ipolterm = PolarTerm(ito, a_pol, a_wght)
        else:
            continue
        if ipolterm in a_pol_terms:
            LOGGER.debug("antonym {:s} is already known".format(
                repr(ipolterm)), file=sys.stderr)
            continue
        else:
            LOGGER.debug("*** adding antonym {:s}".format(
                repr(ipolterm)))
            if ipolterm not in a_pol_terms:
                a_new_terms.add(ipolterm)
                a_pol_terms.add(ipolterm)


def _expand_set(a_germanet, a_polar_terms, a_seeds, a_i,
                a_expanded_syn_rels):
    """Expand set of known polar terms.

    @param a_germanet - GermaNet instance
    @param a_polar_terms - target set of known polar terms
    @param a_seeds - seed polar terms to start from
    @param a_i - weight of new terms
    @param a_expanded_syn_rels - use expanded set of synonymous
      relations

    @return set of new seeds

    """
    new_terms = set()
    for iterm in a_seeds:
        _analyze_term(a_germanet, iterm.term_id,
                      iterm.pol, a_i, new_terms, a_polar_terms,
                      a_expanded_syn_rels)
    return new_terms


def _polterms2lexemes(a_germanet, a_polterms):
    """Convert polar terms to a set of lexemes.

    @param a_germanet - GermaNet instance
    @param a_polterms - set of polar terms

    @return list of 3-tuples: (lexeme, polarity, weight)

    """
    ret = []
    seen_lexes = set()
    for iterm in a_polterms:
        for ilex in a_germanet.lexid2lex[iterm.term_id]:
            if ilex in seen_lexes or iterm.pol == NEUTRAL:
                continue
            seen_lexes.add(ilex)
            ret.append((ilex, iterm.pol, iterm.wght))
    return ret


def hu_liu(a_germanet, a_pos, a_neg, a_neut, a_seed_pos, a_expanded_syn_rels):
    """Extend sentiment lexicons using the  method of Hu and Liu (2004).

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no
      restriction)
    @param a_expanded_syn_rels - use expanded set of synonymous
      relations

    @return \c void

    """
    if a_seed_pos == "none":
        a_seed_pos = None
    polar_terms = set()
    new_terms = set()
    # initialize seed sets
    polar_terms |= _lexemes2polterms(a_germanet, a_pos,
                                     POSITIVE, a_seed_pos)
    n_pos = len(polar_terms)
    print("# of positive lex ids =", n_pos, file=sys.stderr)

    polar_terms |= _lexemes2polterms(a_germanet, a_neg,
                                     NEGATIVE, a_seed_pos)
    n_neg = len(polar_terms) - n_pos
    print("# of negative lex ids =", n_neg, file=sys.stderr)

    polar_terms |= _lexemes2polterms(a_germanet, a_neut,
                                     NEUTRAL, a_seed_pos)
    n_neut = len(polar_terms) - n_pos - n_neg
    print("# of neutral lex ids =", n_neut, file=sys.stderr)

    new_terms |= polar_terms

    i, n = 1., len(new_terms)
    # expand seed sets
    while n:
        i /= 2.
        prev_n = n
        new_terms = _expand_set(a_germanet, polar_terms, new_terms, i,
                                a_expanded_syn_rels)
        n = len(new_terms)
    # convert obtained lex ids back to lexemes
    ret = _polterms2lexemes(a_germanet, polar_terms)
    ret.sort(key=lambda el: el[-1], reverse=True)
    return ret
