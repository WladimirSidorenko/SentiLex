#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
Module for processing Zurich Polarity lexicon

Constants:
LEXICON - default file name of the polarity lexicon
TAB_RE - regexp matching tab separators
POSITIVE - symbolic representation of the positive class
NEGATIVE - symbolic representation of the negative class
NEUTRAL - symbolic representation of the neutral class
KNOWN_CLASSES - allowed polarity classes
COMMENT_RE - regexp matching line comments
EQUAL_RE - regexp matching the equal sign
SPACE_RE - regexp matching whitespaces
ENCODING - deault encoding of lexicon files

Classes:
ZRCH - main interface for the Zurich Polarity lexicon

"""

##################################################################
# Classes
from __future__ import print_function
from generate_lexicon import normalize

import codecs
import os
import re
import sys

##################################################################
# Constants
LEXICON = "german.lex"
POSITIVE = "POS"
NEGATIVE = "NEG"
NEUTRAL = "NEU"
EQUAL_RE = re.compile("\s*=\s*")
COMMENT_RE = re.compile("%%")
SPACE_RE = re.compile("\s+")
KNOWN_CLASSES = set(["NEG", "POS", "NEU", "SHI", "INT"])
ENCODING = "utf-8"

##################################################################
# Classes
class ZRCH(object):
    """
    Class for reading and processing Zurich polarity lexicon

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
        ## dictionary of negative sentiment words
        self.negative = dict()
        ## dictionary of neutral sentiment words
        self.neutral = dict()
        ## dictionary of positive sentiment words
        self.positive = dict()
        self._read_dict(ilex)

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

    def _read_dict(self, a_fname):
        """
        Class constructor

        @param a_fname - source file to read from

        @return \c void
        """
        score = 0.0
        trg_dict = None; ivalue = None
        iform = iclass_score = iclass = iscore = ""
        with codecs.open(a_fname, 'r', encoding = ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not iline or COMMENT_RE.match(iline):
                    continue
                iform, iclass_score = SPACE_RE.split(iline)[:2]
                iclass, iscore = EQUAL_RE.split(iclass_score)
                assert iclass in KNOWN_CLASSES, \
                    "Unknown polarity class: {:s}".format(iclass).encode(ENCODING)
                score = float(iscore)
                iform = normalize(iform)
                if iclass == POSITIVE:
                    trg_dict = self.positive
                elif iclass == NEGATIVE:
                    trg_dict = self.negative
                elif iclass == NEUTRAL:
                    trg_dict = self.neutral
                else:
                    continue

                ivalue = ("NONE", score, iclass)
                if iform in trg_dict:
                    if abs(trg_dict[iform][1]) < abs(score):
                        trg_dict[iform] = ivalue
                else:
                    trg_dict[iform] = ivalue
