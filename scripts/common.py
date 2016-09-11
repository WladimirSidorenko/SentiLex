#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module comprising common utilities for lexicon generation.

"""

##################################################################
# Imports
from germanet import normalize
from tokenizer import Tokenizer

import re

##################################################################
# Constants
ESC_CHAR = ''
SENT_END_RE = re.compile("\s*<\s*sentence\s*/\s*>\s*$")
TAB_RE = re.compile(' *\t+ *')
# the `#' and `,' characters were only added later for the NWE and corpus
# methods, when I discovered that hashtags were skipped during processing that
# hastags were skipped from processing
WORD_RE = re.compile('^[-#.,\w]+$')
ENCODING = "utf-8"

# not sure whether "has_hypernym" should be added to SYNRELS
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"
INFORMATIVE_TAGS = set(["AD", "FM", "NE", "NN", "VV"])
POL2OPPOSITE = {POSITIVE: NEGATIVE, NEGATIVE: POSITIVE}

NEGATORS = set(["nicht", "keine", "kein", "keines", "keinem", "keinen"])
STOP_WORDS = set()
FORM2LEMMA = dict()

ANTONYM = "has_antonym"
SYNONYM = "has_synonym"
ANTIRELS = set([ANTONYM])
# excluded `is_related_to' as it connected `Form' and `unförmig'
SYNRELS = set(["has_participle", "has_pertainym",
               "has_hyponym", "entails", "is_entailed_by"])

TOKENIZER = Tokenizer()


##################################################################
# Imports
def lemmatize(a_form, a_prune=True):
    """
    Convert word form to its lemma

    @param a_form - word form for which we should obtain lemma
    @param a_prune - flag indicating whether uninformative words
                    should be pruned

    @return lemma of the word
    """
    a_form = normalize(a_form)
    if a_prune and a_form in STOP_WORDS:
        return None
    if a_form in FORM2LEMMA:
        return FORM2LEMMA[a_form]
    return a_form


def check_word(a_word):
    """Check if given word forms a valid lexeme

    @param a_word - word to be checked

    @return \c True if word forms a valid lexeme, \c False otherwise

    """
    return WORD_RE.match(a_word) and all(ord(c) < 256 for c in a_word)
