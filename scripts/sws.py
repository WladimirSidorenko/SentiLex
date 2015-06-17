#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
Module for processing SentiWS lexicon

Constants:

Classes:
SWS - main interface to the SentiWS lexicon

"""

##################################################################
# Classes
from .generate_lexicon import normalize

import os
import re

##################################################################
# Constants
SWS_NEGATIVE = "SentiWS_v1.8c_Negative.txt"
SWS_POSITIVE = "SentiWS_v1.8c_Positive.txt"
TAB_RE = re.compile(" *(?:\t *)+")
COMMA_RE = re.compile(" *(?:, *)+")
BAR_RE = re.compile(" *[|] *")
ENCODING = "utf-8"

##################################################################
# Classes
class SWS(object):
    """
    Class for reading and processing SentiWS lexicon data

    Instance variables:
    negative - dictionary of negative Sentiment words
    positive - dictionary of positive Sentiment words

    Methods:
    check_word - check if given word is present in the lexicon
    check_word_tag - check if given word is present in the lexicon with specified tag
    """

    def __init__(self, a_dir):
        """
        Class constructor

        @param a_dir - directory containing SentiWS lexicon
        """
        if not os.path.isdir(a_dir) or not os.access(a_dir, os.R_OK):
            raise RuntimeError("Cannot acess directory {:s}".format(a_dir))
        inegative = os.path.join(a_dir, SWS_NEGATIVE)
        ipositive = os.path.join(a_dir, SWS_POSITIVE)
        if not os.path.exists(inegative) or not os.path.exists(ipositive):
            raise RuntimeError("SentiWS files not found in directory {:s}".format(a_dir))
        # initialize instance variables
        self.negative = _read_dict(inegative)
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

    def _read_dict(self, a_dict):
        """
        Class constructor

        @param a_dict - file containing dictionary entries

        @return dictionary read
        """
        ret = dict()
        tags = []
        iword = iscore = iforms = ivalue = None
        word = ""; tag = ""; forms = []; score = 0.
        with codecs.open(a_dict, ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not line:
                    continue
                iword, iscore, iforms = TAB_RE.split(iline)
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
        return ret
