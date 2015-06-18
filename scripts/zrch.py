#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
Module for processing Zurich Polarity lexicon

Constants:

Classes:
ZRCH - main interface for the Zurich Polarity lexicon

"""

##################################################################
# Classes
from .generate_lexicon import normalize

import os
import re

##################################################################
# Constants
LEXICON = "german.lex"
POSITIVE = "POS"
NEGATIVE = "NEG"
NEUTRAL = "NEU"
EQUAL_RE = re.compile("\s*=\s*")
COMMENT_RE = re.compile("%%")
ENCODING = "utf-8"

##################################################################
# Classes
class ZRCH(object):
    """
    Class for reading and processing Zurich Polarity lexicon

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
        ilex = os.path.join(a_dir, LEXICON)
        if not os.path.exists(ilex):
            raise RuntimeError("Lexicon file not found in directory {:s}".format(a_dir))
        # initialize instance variables
        self.negative = dict(); self.neutral = dict(); self.positive = dict()
        _read_dict(inegative, NEGATIVE, self.negative)
        _read_dict(ineutral, NEUTRAL, self.neutral)
        _read_dict(ineutral, POSITIVE, self.positive)

    def check_word(self, a_word):
        """
        Check if given word is present in the lexicon

        @param a_word - word to be checked

        @return list of word's tags and scores found in dictionaries
        """
        ret = []
        iword = normalize(a_word)
        if iword in self.negative:
            ret += self.negative[iword]
        if iword in self.positive:
            ret += self.positive[iword]
        if iword in self.neutral:
            ret += self.neutral[iword]
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
        with codecs.open(a_fname, ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not line:
                    continue
                iform, ilemma, itag, iclass, iscores, _ = TAB_RE.split(iline)
                assert a_class == iclass, \
                    "Mismatching classes: '{:s}' vs. '{:s}'".format(a_class, iclass)
                iscores = [0.0 if s == '-' else float(s) for s in SLASH_RE.split(iscores)][CLASS2IDX[a_class]]
                if itag == "AD":
                    itags = ["ADJA", "ADJD"]
                else:
                    itags = [itag]
                iform = normalize(iform); ilemma = normalize(ilemma)
                iforms = (iform, ilemma)
                for itag in itags:
                    ivalue = (itag, score, iclass)
                    for iform in iforms:
                        if iform in ret:
                            a_dict[iform].append(ivalue)
                        else:
                            a_dict[iform] = [ivalue]
