#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""
Script for generating sentiment lexicons

USAGE:
generate_lexicon.py [OPTIONS] [INPUT_FILES]

"""

##################################################################
# Imports
from __future__ import unicode_literals
from germanet import Germanet
from gpc import POSITIVE, NEGATIVE, NEUTRAL

import argparse
import re
import sys

##################################################################
# Imports
TAKAMURA = "takamura"
ESULI = "esuli"
W2V = "w2v"
TAB_RE = re.compile(' *\t+ *')
SPACE_RE = re.compile('(?:\t| \s)+')
ENCODING = "utf-8"
GNET_DIR = "germanet_dir"
POSITIVE = "positive"
POS_SET = set()                 # set of positive terms
NEGATIVE = "negative"
NEG_SET = set()                 # set of negative terms
NEUTRAL = "neutral"
NEUT_SET = set()                # set of neutral terms
SZET_RE = re.compile('ß', re.U)

##################################################################
# Main
def normalize(a_string):
    """
    Lowercase string and replace multiple whitespaces.

    @param a_string - string to normalize

    @return normalized string version
    """
    return SZET_RE.sub("ss", SPACE_RE.sub(' ', a_string.lower()))

def _read_set(a_fname):
    """
    Read initial seed set of terms.

    @param a_fname - name of input file containing terms

    @return \c void
    """
    global POS_SET, NEG_SET, NEUT_SET
    fields = []
    with codecs.open(a_fname, 'r', encoding = ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not line:
                continue
            fields = TAB_RE.split(iline)
            if fields[-1] == POSITIVE:
                POS_SET.add(normalize(fields[0]))
            elif fields[-1] == NEGATIVE:
                NEG_SET.add(normalize(fields[0]))
            elif fields[-1] == NEUTRAL:
                NEUT_SET.add(normalize(fields[0]))
            else:
                raise RuntimeError("Unknown field specification: {:s}".format(fields[-1]))

def _lexeme2synset(a_lexemes):
    """
    Extract all synsets containing lexemes

    @param a_lexemes - set of lexemes for which to extract the synsets

    @return set of synset id's which contain lexemes
    """
    ret = set()
    for ilex in a_lexemes:
        for isyn_id in a_germanet.lex2synids.get(ilex, []):
            ret.add(isyn_id)
    return ret

def esuli_sebastiani(a_germanet, a_N, a_pos, a_neg, a_neut):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_germanet - GermaNet instance
    @param a_N - number of terms to extract
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms
    @param a_neut - initial set of neutral terms

    @return \c 0 on success, non-\c 0 otherwise
    """
    global POS_SET, NEG_SET, NEUT_SET
    # convert lexemes to synsets
    ipos = _lexeme2synset(POS_SET); ineg = _lexeme2synset(NEG_SET); ineut = _lexeme2synset(NEUT_SET)
    # train classifier on each of the sets
    for ipos in a_pos:
        for isyn_id in a_germanet.lex2synids.get(ipos, []):
            for itrg_syn_id, irelname in a_germanet.relations.get(isyn_id, [(None, None)]):
                if irelname in SYNRELS:
                    for ilex in a_germanet.synid2lex[itrg_syn_id]:
                        pos_candidates.add(ilex)
                elif irelname in ANTIRELS:
                    for ilex in a_germanet.synid2lex[itrg_syn_id]:
                        neg_candidates.add(ilex)
    for ipos in a_pos:
        for isyn_id in a_germanet.lex2synids.get(ipos, []):
            for itrg_syn_id, irelname in a_germanet.relations.get(isyn_id, [(None, None)]):
                if irelname in SYNRELS:
                    for ilex in a_germanet.synid2lex[itrg_syn_id]:
                        pos_candidates.add(ilex)
                elif irelname in ANTIRELS:
                    for ilex in a_germanet.synid2lex[itrg_syn_id]:
                        neg_candidates.add(ilex)
    # return the union of three sets
    return a_pos | a_neg | a_neut

def takamura(a_gnet_dir, a_N, a_pos, a_neg, a_neut):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_gnet_dir - directory containing GermaNet files
    @param a_N - number of terms to extract
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms
    @param a_neut - initial set of neutral terms

    @return \c 0 on success, non-\c 0 otherwise
    """
    ret = set()
    return ret

def main(a_argv):
    """
    Main method for generating sentiment lexicons

    @param a_argv - command-line arguments

    @return \c 0 on success, non-\c 0 otherwise
    """
    argparser = argparse.ArgumentParser(description = """Script for \
generating sentiment lexicons.""")
    # add type-specific subparsers
    subparsers = argparser.add_subparsers(help = "lexicon expansion method to use", dest = "dmethod")

    subparser_takamura = subparsers.add_parser(TAKAMURA, help = "Ising spin model (Takamura, 2005)")
    subparser_takamura.add_argument(GNET_DIR, help = "directory containing GermaNet files")
    subparser_takamura.add_argument("corpus_dir", help = "directory containing raw corpus files")
    subparser_takamura.add_argument("N", help = "final number of additional terms to extract")

    subparser_esuli = subparsers.add_parser(ESULI, help = "SentiWordNet model (Esuli and Sebastiani, 2005)")
    subparser_esuli.add_argument(GNET_DIR, help = "directory containing GermaNet files")
    subparser_esuli.add_argument("N", help = "number of expansion iterations")

    subparser_w2v = subparsers.add_parser(W2V, help = "word2vec model (Mikolov, 2013)")
    subparser_w2v.add_argument("N", help = "final number of terms to extract")

    argparser.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    args = argparser.parse_args(a_argv)

    # initialize GermaNet, if needed
    igermanet = None
    if GNET_DIR in args:
        igermanet = Germanet(getattr(args, GNET_DIR))

    # obtain lists of conjoined terms, if needed

    # read seed sets
    _read_set(args.seed_set)

    # apply requested method
    if args.dmethod == ESULI:
        new_set = esuli_sebastiani(igermanet, N, POS_SET, NEG_SET, NEUT_SET)
    elif args.dmethod == TAKAMURA:
        new_set = takamura(igermanet, N, POS_SET, NEG_SET, NEUT_SET)

    for iexpression in sorted(new_set):
        print iexpression.encode(ENCODING)

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
