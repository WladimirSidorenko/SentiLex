#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
Module for processing SentiWS lexicon

Constants:
SWS_NEGATIVE - default name of negative polarity lexicon
SWS_POSITIVE - default name of positive polarity lexicon
TAB_RE - regexp matching tab separators
COMMA_RE - regexp matching comma separators
BAR_RE - regexp matching vertical bars
ENCODING - deault encoding of lexicon files

Classes:
SWS - main interface to the SentiWS lexicon

"""

##################################################################
# Classes
from generate_lexicon import normalize

import codecs
import os
import re

##################################################################
# Constants
NEGATIVE = "negative"
SWS_NEGATIVE = "SentiWS_v1.8c_Negative.txt"
POSITIVE = "positive"
SWS_POSITIVE = "SentiWS_v1.8c_Positive.txt"
NEUTRAL = "neutral"
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
    lemma2forms - dictionary mapping lemmas to forms
    form2lemma - dictionary mapping form to lemma
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
        ## dictionary mapping lemmas to forms
        self.lemma2forms = dict()
        ## dictionary of negative Sentiment words
        self.negative = self._read_dict(inegative, NEGATIVE)
        ## dictionary of positive Sentiment words
        self.positive = self._read_dict(ipositive, POSITIVE)
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

    def _read_dict(self, a_dict, a_class):
        """
        Class constructor

        @param a_dict - file containing dictionary entries
        @param a_class - expected target class of the entries

        @return dictionary read
        """
        ret = dict()
        fields = []; tags = []
        iword = iscore = iforms = ivalue = None
        word = ""; tag = ""; forms = []; score = 0.
        with codecs.open(a_dict, 'r', encoding = ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not iline:
                    continue
                fields = TAB_RE.split(iline)
                iword, iscore = fields[:2]
                if len(fields) == 3:
                    iforms = fields[-1]
                else:
                    iforms = ""
                word, tag = BAR_RE.split(iword)
                score = float(iscore)
                if tag == "ADJX":
                    tags = ["ADJA", "ADJD"]
                else:
                    tags = [tag]
                forms = [normalize(f) for f in COMMA_RE.split(iforms) if f]
                forms.append(normalize(word))
                for itag in tags:
                    ivalue = (itag, score, a_class)
                    for iform in forms:
                        if iform in a_dict:
                            if abs(a_dict[iform][1]) < abs(ivalue[1]):
                                ret[iform] = ivalue
                        else:
                            ret[iform] = ivalue
        return ret
