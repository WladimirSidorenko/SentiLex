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
from .generate_lexicon import normalize

import os
import re

##################################################################
# Constants
GPC_NEGATIVE = "GermanPolarityClues-Negative-21042012.tsv"
GPC_POSITIVE = "GermanPolarityClues-Positive-21042012.tsv"
GPC_NEUTRAL = "GermanPolarityClues-Neutral-21042012.tsv"
TAB_RE = re.compile(" *(?:\t *)+")
SLASH_RE = re.compile(" *(?:/ *)+")
ENCODING = "utf-8"

##################################################################
# Classes
class GPC(object):
    """
    Class for reading and processing German Polarity Clues lexicon

    Instance variables:
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
        self.negative = _read_dict(inegative)
        self.neutral = _read_dict(ineutral)
        self.positive = _read_dict(ipositive)

    def check_word(self, a_word):
        """
        Check if given word is present in the lexicon

        @param a_word - word to be checked

        @return list of word's tags and scores found in dictionaries
        """
        iword = normalize(a_word)
        ret = []
        if iword in self.negative:
            ret += self.negative[iword]
        if iword in self.positive:
            ret += self.positive[iword]
        return ret

    def check_word_tag(self, a_word, a_tag):
        """
        Check if given word is present in the lexicon with the specified tag

        @param a_word - word to be checked
        @param a_tag - tag of the checked word

        @return score for the word with that tag or 0.0 if the pair is not present
        """
        for iscore, itag in self.check_word(a_word):
            if itag == a_tag:
                return iscore
        return 0.0

    def _read_dict(self, a_fname, a_dict):
        """
        Class constructor

        @param a_fname - source file to read from
        @param a_dict - dictionary to be poopulated

        @return \c void
        """
        scores = []
        iform = ilemma = itag = iclass = iscores = ""
        with codecs.open(a_fname, ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not line:
                    continue
                iform, ilemma, itag, _, iscores, _ = TAB_RE.split(iline)
                word, tag = BAR_RE.split(iword)
                score = float(iscore)
                if tag == "ADJX":
                    tags = ["ADJA", "ADJD"]
                else:
                    tags = [tag]
                forms = [normalize(f) for f in COMMA_RE.split(iforms)]
                forms.append(normalize(word))
                for itag in tags:
                    ivalue = (itag, score)
                    for iform in forms:
                        if iform in ret:
                            ret[iform].append(ivalue)
                        else:
                            ret[iform] = [ivalue]
