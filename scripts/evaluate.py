#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Script for evaluating sentiment lexicon on test corpus.

USAGE:
evaluate.py [lemma_file] sentiment_lexicon test_corpus_dir/

"""

##################################################################
# Libraries
import argparse
import re
import sys

##################################################################
# Constants and Variables
LEXICON = dict()

##################################################################
# Methods
def _read_lexicon(a_fname):
    """
    Read sentiment lexicon into global dictionary

    @param a_fname - name of the file containign sentiment lexicon

    @return \c void
    """
    pass

##################################################################
# Arguments
argparser = argparse.ArgumentParser(description = \
                                        """Script for evaluating sentiment lexicon on test corpus.""")
argparser.add_argument("--lemma_file", help = "file containing lemmas of corpus words", type = str)
argparser.add_argument("sentiment_lexicon", help = "sentiment lexicon to test", type = str)
argparser.add_argument("test_corpus_dir", help = \
                           """directory containing sentiment corpus to test against""", type = str)
args = argparser.parse_args()

