#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""
Script for generating sentiment lexicons

USAGE:
generate_lexicon.py [OPTIONS] [INPUT_FILES]

"""

##################################################################
# Imports
from germanet import Germanet

import argparse
import sys

##################################################################
# Imports
TAKAMURA = "takamura"
ESULI = "esuli"
W2V = "w2v"
GNET_DIR = "germanet_dir"

##################################################################
# Main
def esuli_sebastiani(a_gnet_dir, a_N, a_pos, a_neg, a_neut):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_gnet_dir - directory containing GermaNet files
    @param a_N - number of terms to extract
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms
    @param a_neut - initial set of neutral terms

    @return \c 0 on success, non-\c 0 otherwise
    """
    return 0

def main(a_argv):
    """
    Main method for generating sentiment lexicons

    @param a_argv - command-line arguments

    @return \c 0 on success, non-\c 0 otherwise
    """
    argparser = argparse.ArgumentParser(description = """Script for \
generating sentiment lexicons.""")
    # add type-specific subparsers
    subparsers = parser.add_subparsers(help = "lexicon expansion method to use", dest = "dmethod")

    parser_takamura = subparsers.add_parser(TAKAMURA, help = "Ising spin model (Takamura, 2005)")
    parser_takamura.add_argument(GNET_DIR, help = "directory containing GermaNet files")
    parser_takamura.add_argument("corpus_dir", help = "directory containing raw corpus files")
    parser_takamura.add_argument("N", help = "final number of terms to extract")

    parser_esuli = subparsers.add_parser(ESULI, help = "SentiWordNet model (Esuli and Sebastiani, 2005)")
    parser_esuli.add_argument(GNET_DIR, help = "directory containing GermaNet files")
    parser_esuli.add_argument("N", help = "number of expansion iterations")

    parser_w2v = subparsers.add_parser(W2V, help = "word2vec model (Mikolov, 2013)")
    parser_w2v.add_argument("N", help = "final number of terms to extract")

    argparser.add_argument("seed_pos", help = "initial seed set of positive terms")
    argparser.add_argument("seed_neg", help = "initial seed set of negative terms")
    argparser.add_argument("seed_neut", help = "initial seed set of neutral terms")

    args = argparser.parse_args(a_argv)

    # initialize GermaNet, if needed
    if GNET_DIR in args:
        igermanet = Germanet(getattr(args, GNET_DIR))
    else:
        igermanet = None

    # obtain lists of conjoined terms, if needed

    # read seed sets

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
