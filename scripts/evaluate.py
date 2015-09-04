#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Script for evaluating sentiment lexicon on test corpus.

USAGE:
evaluate.py [lemma_file] sentiment_lexicon test_corpus_dir/

"""

##################################################################
# Libraries
from trie import Trie

import argparse
import codecs
import re
import sys
import xml.etree.ElementTree as ET

##################################################################
# Constants and Variables
ENCODING = "utf-8"              # default encoding of input files
POSITIVE = 0                    # positive subjectivity
NEGATIVE = 1                    # negative subjectivity
COMMENT_RE = re.compile("(?:\A|\s+)#")
TAB_RE = re.compile("[ ]*\t+[ ]*")

##################################################################
# Methods
def read_file(a_lexicon, a_fname, a_insert, a_enc = ENCODING):
    """
    General method for reading tab-separated files

    @param a_lexicon - lexicon to be populated
    @param a_fname - name of the file containign sentiment lexicon
    @param a_insert - custom insert function
    @param a_enc - encoding of the input file

    @return \c void
    """
    item1 = item2 = None
    with codecs.open(a_fname, 'r', a_enc) as ifile:
        for iline in ifile:
            iline = iline.strip()
            iline = COMMENT_RE.sub("", iline)
            if not iline:
                continue
            try:
                item1, item2 = TAB_RE.split(iline)
            except ValueError:
                print >> sys.stderr, "Invalid line format: '{:s}'".format(iline).encode(a_enc)
                raise
            a_insert(a_lexicon, item1, item2)

def eval_lexicon(a_lexicon, a_corpus_files):
    """
    Evaluate sentiment lexicon on a real corpus

    @param a_lexicon - lexicon test
    @param a_corpus_files - list of .mmax project files

    @return 6-tuple with macro- and micro-averaged precision, recall, and F-measure
    """
    idoc = None
    for ifname in a_corpus_files:
        idoc = ET.parse(ifname).getroot()
        
        for ielem in idoc.iter():
            print repr(ielem)

def main(argv):
    """
    Main method for estimating quality of a sentiment lexicon

    @param argv - CLI arguments

    @return 0 on success, non-0 otherwise
    """
    # parse arguments
    argparser = argparse.ArgumentParser(description = \
                                        """Script for evaluating sentiment lexicon on test corpus.""")
    argparser.add_argument("-e", "--encoding", help = "encoding of input files", type = str, \
                               default = ENCODING)
    argparser.add_argument("--lemma-file", help = "file containing lemmas of corpus words", \
                               type = str)
    argparser.add_argument("sentiment_lexicon", help = "sentiment lexicon to test", type = str)
    argparser.add_argument("corpus_base_dir", help = \
                           "directory containing word files of sentiment corpus in MMAX format", \
                               type = str)
    argparser.add_argument("corpus_anno_dir", help = \
                           "directory containing annotation files of sentiment corpus in MMAX format", \
                               type = str)
    args = argparser.parse_args(argv)
    # read-in lexicon
    ilex = Trie()
    read_file(ilex, args.sentiment_lexicon, a_insert = lambda lex, wrd, cls: lex.add(wrd, cls))
    form2lemma = dict()
    if args.lemma_file is not None:
        read_file(form2lemma, args.lemma_file, a_insert = \
                      lambda lex, form, lemma: lex.setdefault(form, lemma))
    # evaluate it on corpus
    eval_lexicon(ilex, args.corpus_files)

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
