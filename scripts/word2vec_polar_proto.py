#!/usr/bin/env python
# -*- coding: utf-8; -*-

"""Python prototype of polarity-aware `word2vec`.

USAGE:
word2vec_po;ar_proto.py [OPTIONS] seed_file input_file(s)

OPTIONS:
-d|--dimension -- dimensionality of learned vectors (-1)
-m|--min-freq -- minimal frequency of word to be considered
-w|--window -- size of context window

"""

##################################################################
# Imports
from __future__ import print_function
from evaluate import ENCODING, TAB_RE
from trie import normalize_string

from gensim.models import Word2Vec
import argparse
import codecs
import sys

##################################################################
# Classes
class IFiles(Object):
    """Auxiliary class for iterating over input lines.

    Instance variables:
    ifiles - list of input file names

    Methods:
    __init__ - class constructor
    __iter__ - iterator

    """

    def __init__(self, a_fnames):
        """Class constructor

        @param a_fnames - list of input files

        """
        self.ifiles = a_fnames or ["-"]

    def __iter__(self):
        """Iterator over input lines

        @return iterator over input lines

        """
        for ifname in self.ifiles:
            with (codecs.open(ifname, 'r', ENCODING) \
                      if ifname != '-' else sys.stdin) as ifile:
                for iline in ifile:
                    yield iline.strip()
        raise StopIteration

class PolarWord2Vec(Word2Vec):
    """Polarity-aware `word2vec` model

    This class adds following additional members:


    This subclass overwrites following methods:
    """
    pass

##################################################################
# Variables and Constants
POL2WSCORE = {"positive": 1., "negative": -1.}

##################################################################
# Methods
def main(argv):
    """Main method for generating polarity-aware word vectors

    @param argv - CLI arguments

    @return 0 on succes, non-0 otherwise

    """
    # parse arguments
    argparser = argparse.ArgumentParser(description = \
                                        "Python prototype of polarity-aware `word2vec`.")
    argparser.add_argument("-d", "--dimension", help = "dimensionality of learned vectors (-1)", \
                               type = int, default = 400)
    argparser.add_argument("-m", "--min-freq", help = "minimal frequency of word to be considered", \
                               type = int, default = 4)
    argparser.add_argument("-w", "--window", help = "size of context window", \
                               type = int, default = 5)
    argparser.add_argument("sentiment_lexicon", help = "sentiment lexicon to add", type = str)
    argparser.add_argument("input_files", help = "input files to sample vectors from", \
                               nargs = '*')
    args = argparser.parse_args(argv)

    # read polarity lexicon
    word2pol = dict()
    ifields = []
    with codecs.open(args.sentiment_lexicon, 'r', encoding = ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            ifields = TAB_RE.split(iline)
            word2pol[normalize_string(ifields[0])] = POL2WSCORE[ifields[-1]]

    # train word vectors
    w2v = Word2Vec(IFiles(args.input_files), size = args.dimension, window = args.window, \
                       min_count = args.min_count)

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
