#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
Module for processing German Polarity Clues lexicon

Constants:

Classes:
GPC - main interface for the German Polarity Clues lexicon

"""

##################################################################
# Classes
from __future__ import print_function
from generate_lexicon import normalize

import codecs
import os
import re

##################################################################
# Constants
DELIM = '\t'
NEGATIVE = "negative"
GPC_NEGATIVE = "GermanPolarityClues-Negative-21042012.tsv"
POSITIVE = "positive"
GPC_POSITIVE = "GermanPolarityClues-Positive-21042012.tsv"
NEUTRAL = "neutral"
GPC_NEUTRAL = "GermanPolarityClues-Neutral-21042012.tsv"
CLASS2IDX = {POSITIVE: 0, NEGATIVE: 1, NEUTRAL: 2}
TAB_RE = re.compile(" *(?:\t *)+")
SLASH_RE = re.compile(" *(?:/ *)+")
ENCODING = "utf-8"

##################################################################
# Classes
class GPC(object):
    """
    Class for reading and processing German Polarity Clues lexicon

    Instance variables:
    lemma2forms - dictionary mapping lemmas to forms
    form2lemma - dictionary mapping form to lemma
    negative - dictionary of negative sentiment words
    positive - dictionary of positive sentiment words
    neutral - dictionary of neutral words

    Methods:
    check_word - check if given word is present in the lexicon
    check_word_tag - check if given word is present in the lexicon with specified tag
    """

    def __init__(self, a_dir):
        """
        Class constructor

        @param a_dir - directory containing GPC lexicon
        """
        if not os.path.isdir(a_dir) or not os.access(a_dir, os.R_OK):
            raise RuntimeError("Cannot acess directory {:s}".format(a_dir))
        inegative = os.path.join(a_dir, GPC_NEGATIVE)
        ineutral = os.path.join(a_dir, GPC_NEUTRAL)
        ipositive = os.path.join(a_dir, GPC_POSITIVE)
        if not os.path.exists(inegative) or not os.path.exists(ipositive) or \
                not os.path.exists(ineutral):
            raise RuntimeError("GPC files not found in directory {:s}".format(a_dir))
        # initialize instance variables
        ## dictionary mapping lemmas to forms
        self.lemma2forms = dict()
        ## dictionary of negative sentiment words
        self.negative = dict(); self._read_dict(inegative, NEGATIVE, self.negative)
        ## dictionary of neutral words
        self.neutral = dict(); self._read_dict(ineutral, NEUTRAL, self.neutral)
        ## dictionary of positive sentiment words
        self.positive = dict(); self._read_dict(ipositive, POSITIVE, self.positive)
        ## dictionary mapping form to lemma
        self.form2lemma = {f: lemma for lemma, forms in self.lemma2forms.iteritems() \
                               for f in forms}

    def check_word(self, a_word):
        """
        Check if given word is present in the lexicon

        @param a_word - word to be checked

        @return list of word's tags and scores found in dictionaries
        """
        ret = []
        iword = normalize(a_word)
        if iword in self.negative:
            ret.append(self.negative[iword])
        if iword in self.positive:
            ret.append(self.positive[iword])
        if iword in self.neutral:
            ret.append(self.neutral[iword])
        return ret

    def check_word_tag(self, a_word, a_tag):
        """
        Check if given word is present in the lexicon with the specified tag

        @param a_word - word to be checked
        @param a_tag - tag of the checked word

        @return score for the word with that tag or 0.0 if the pair is not present
        """
        for iscore, itag, iclass in self.check_word(a_word):
            if itag == a_tag:
                return (iscore, iclass)
        return (0.0, NEUTRAL)

    def _read_dict(self, a_fname, a_class, a_dict):
        """
        Class constructor

        @param a_fname - source file to read from
        @param a_class - expected target class of the entries
        @param a_dict - dictionary to be populated

        @return \c void
        """
        scores = []; itags = []; iforms = []
        iform = ilemma = tag = iclass = iscores = ""
        with codecs.open(a_fname, 'r', encoding = ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not iline:
                    continue
                iform, ilemma, itag, iclass, iscores, _ = TAB_RE.split(iline)
                assert a_class == iclass, \
                    "Mismatching classes: '{:s}' vs. '{:s}'".format(a_class, iclass)
                score = SLASH_RE.split(iscores)[CLASS2IDX[a_class]]
                score = 0.0 if score == '-' else float(score)
                if itag == "AD":
                    itags = ["ADJA", "ADJD"]
                else:
                    itags = [itag]
                iform = normalize(iform); ilemma = normalize(ilemma)
                if ilemma in self.lemma2forms:
                    self.lemma2forms[ilemma].update([iform])
                else:
                    self.lemma2forms[ilemma] = set([iform])
                if ilemma not in a_dict:
                    iforms = set([iform, ilemma])
                else:
                    iforms = [iform]
                for itag in itags:
                    ivalue = (itag, score, iclass)
                    for iform in iforms:
                        if iform in a_dict:
                            if abs(a_dict[iform][1]) < abs(ivalue[1]):
                                a_dict[iform] = ivalue
                        else:
                            a_dict[iform] = ivalue

##################################################################
# Main
if __name__ == "__main__":
    # process arguments
    import argparse
    argparser = argparse.ArgumentParser(description = "Merge terms from GPC into a single TSV file.")
    argparser.add_argument("gpc_dir", help = "directory containing German Polarity Clues")
    args = argparser.parse_args()

    # initialize dictionaries
    gpc = GPC(args.gpc_dir)

    # output entries in TSV format
    pos_set = set(gpc.positive.keys())
    neg_set = set(gpc.negative.keys())
    for iset, iclass in ((pos_set, POSITIVE), (neg_set, NEGATIVE)):
        for iword in sorted(iset):
            print((iword + DELIM + iclass).encode(ENCODING))
