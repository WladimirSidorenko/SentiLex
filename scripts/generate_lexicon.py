#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Script for generating sentiment lexica.

USAGE:
generate_lexicon.py [OPTIONS] [INPUT_FILES]

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import lemmatize, _lemmatize, ANTIRELS, SYNRELS, TOKENIZER, \
    POSITIVE, NEGATIVE, NEUTRAL

from awdallah import awdallah
from blair_goldensohn import blair_goldensohn
from esuli_sebastiani import esuli_sebastiani
from hu_liu import hu_liu
from germanet import Germanet, normalize, POS
from ising import Ising, ITEM_IDX, WGHT_IDX, HAS_FXD_WGHT, FXD_WGHT_IDX

from itertools import chain, combinations
import argparse
import codecs
from math import floor, ceil, isnan
import numpy as np
import os
import re
import sys
import string

##################################################################
# Imports
GNET_DIR = "germanet_dir"
CC_FILE = "cc_file"

VERBOSE = False

AWDALLAH = "awdallah"
BG = "blair-goldensohn"
ESULI = "esuli"
HU = "hu"
KIM = "kim"
RAO = "rao"
TAKAMURA = "takamura"
W2V = "w2v"

NEGATORS = set(["nicht", "keine", "kein", "keines", "keinem", "keinen"])

W_DELIM_RE = re.compile('(?:\s|{:s})+'.format(
    '|'.join([re.escape(c) for c in string.punctuation])))
WORD_RE = re.compile('^[-.\w]+$')
TAB_RE = re.compile(' *\t+ *')
ENCODING = "utf-8"
POS_SET = set()                 # set of positive terms
NEG_SET = set()                 # set of negative terms
NEUT_SET = set()                # set of neutral terms

INFORMATIVE_TAGS = set(["AD", "FM", "NE", "NN", "VV"])
STOP_WORDS = set()
FORM2LEMMA = dict()


##################################################################
# Main
def _add_cmn_opts(a_parser):
    """Add options common to all option parsers.

    @param a_parser - argument parser to add options to

    @return \c void

    """
    a_parser.add_argument("--seed-pos",
                          help="part-of-speech of seed synsets"
                          " ('none' for no restriction)",
                          choices=["none"] + [p[:-1] for p in POS],
                          default="none"
                          )
    a_parser.add_argument("--form2lemma", "-l",
                          help="file containing form-lemma"
                          " correspondences", type=str)
    a_parser.add_argument("seed_set",
                          help="initial seed set of positive,"
                          " negative, and neutral terms")
    a_parser.add_argument(GNET_DIR,
                          help="directory containing GermaNet files")


def _get_form2lemma(a_fname):
    """Read file containing form/lemma correspodences

    @param a_fname - name of input file

    @return \c void (correspondences are read into global variables)

    """
    global STOP_WORDS, FORM2LEMMA

    if not os.path.isfile(a_fname) or not os.access(a_fname, os.R_OK):
        raise RuntimeError("Cannot read from file '{:s}'".format())

    iform = itag = ilemma = ""
    with codecs.open(a_fname, 'r', encoding=ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            iform, itag, ilemma = TAB_RE.split(iline)
            iform = normalize(iform)
            if len(itag) > 1 and itag[:2] in INFORMATIVE_TAGS:
                FORM2LEMMA[iform] = normalize(ilemma)
            else:
                STOP_WORDS.add(iform)


def _read_set(a_fname):
    """Read initial seed set of terms.

    @param a_fname - name of input file containing terms

    @return \c void

    """
    global POS_SET, NEG_SET, NEUT_SET
    fields = []
    with codecs.open(a_fname, 'r',
                     encoding=ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            fields = TAB_RE.split(iline)
            if fields[-1] == POSITIVE:
                POS_SET.add(normalize(fields[0]))
            elif fields[-1] == NEGATIVE:
                NEG_SET.add(normalize(fields[0]))
            elif fields[-1] == NEUTRAL:
                NEUT_SET.add(normalize(fields[0]))
            else:
                raise RuntimeError(
                    "Unknown field specification: {:s}".format(fields[-1]))


def _check_word(a_word):
    """Check if given word forms a valid lexeme

    @param a_word - word to be checked

    @return \c True if word forms a valid lexeme, \c False otherwise

    """
    return WORD_RE.match(a_word) and all(ord(c) < 256 for c in a_word)


def _tkm_add_germanet(ising, a_germanet):
    """
    Add lexical nodes from GermaNet to the Ising spin model

    @param a_ising - instance of the Ising spin model
    @param a_germanet - GermaNet instance

    @return \c void
    """
    # add all lemmas from the `FORM2LEMMA` dictionary
    for ilemma in FORM2LEMMA.itervalues():
        if ilemma not in STOP_WORDS:
            ising.add_node(ilemma)
    # add all lemmas from synsets
    for ilexid in a_germanet.lexid2synids.iterkeys():
        for ilex in a_germanet.lexid2lex[ilexid]:
            ising.add_node(ilex)
    # establish links between synset words and lemmas appearing in
    # examples and definitions
    def_lexemes = []
    negation_seen = False
    for isynid, (idef, iexamples) in a_germanet.synid2defexmp.iteritems():
        def_lexemes = [lemmatize(iword, a_prune=False)
                       for itxt in chain(idef, iexamples)
                       for iword in TOKENIZER.tokenize(itxt)]
        def_lexemes = [ilexeme
                       for ilexeme in def_lexemes
                       if ilexeme
                       and ising.item2nid.get(ilexeme, None) is not None]
        if def_lexemes:
            negation_seen = False
            for idef_lex in def_lexemes:
                if idef_lex in NEGATORS:
                    negation_seen = True
                    continue
                elif idef_lex in STOP_WORDS:
                    continue
                for ilexid in a_germanet.synid2lexids[isynid]:
                    for ilex in a_germanet.lexid2lex[ilexid]:
                        ising.add_edge(ilex,
                                       idef_lex, -1. if negation_seen else 1.)
    # establish links between synset lemmas based on the lexical
    # relations
    iwght = 1.
    lemmas1 = lemmas2 = None
    for ifrom, irelset in a_germanet.lex_relations.iteritems():
        lemmas1 = a_germanet.lexid2lex.get(ifrom)
        assert lemmas1 is not None, "No lemma found for id {:s}".format(ifrom)
        for ito, irel in irelset:
            lemmas2 = a_germanet.lexid2lex.get(ito)
            assert lemmas2 is not None, \
                "No lemma found for id {:s}".format(ito)
            if irel in SYNRELS:
                iwght = 1.
            elif irel in ANTIRELS:
                iwght = -1.
            else:
                continue
            for ilemma1 in lemmas1:
                for ilemma2 in lemmas2:
                    ising.add_edge(ilemma1, ilemma2, iwght)
    # establish links between synset lemmas based on the con relations
    for ifrom, irelset in a_germanet.con_relations.iteritems():
        # iterate over all lexemes pertaining to the first synset
        for ilex_id1 in a_germanet.synid2lexids[ifrom]:
            lemmas1 = a_germanet.lexid2lex.get(ilex_id1)
            assert lemmas1 is not None, \
                "No lemma found for id {:s}".format(ifrom)
            # iterate over target synsets and their respective relations
            for ito, irel in irelset:
                if irel in SYNRELS:
                    iwght = 1.
                elif irel in ANTIRELS:
                    iwght = -1.
                else:
                    continue
                # iterate over all lexemes pertaining to the second synset
                for ilex_id2 in a_germanet.synid2lexids[ito]:
                    lemmas2 = a_germanet.lexid2lex.get(ilex_id2)
                    assert lemmas2 is not None, \
                        "No lemma found for id {:s}".format(ito)
                    for ilemma1 in lemmas1:
                        for ilemma2 in lemmas2:
                            ising.add_edge(ilemma1, ilemma2, iwght)
    # establish links between lemmas which pertain to the same synset
    ilexemes = set()
    for ilex_ids in a_germanet.synid2lexids.itervalues():
        ilexemes = set([ilex
                        for ilex_id in ilex_ids
                        for ilex in a_germanet.lexid2lex[ilex_id]])
        # generate all possible (n choose 2) combinations of lexemes
        # and put links between them
        for ilemma1, ilemma2 in combinations(ilexemes, 2):
            ising.add_edge(ilemma1, ilemma2, 1.)


def _tkm_add_corpus(ising, a_cc_file):
    """Add lexical nodes from corpus to the Ising spin model

    @param a_ising - instance of the Ising spin model
    @param a_cc_file - file containing conjoined word pairs extracted from
      corpus

    @return \c void

    """
    ifields = []
    iwght = 1.
    ilemma1 = ilemma2 = ""
    with codecs.open(a_cc_file, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            ifields = TAB_RE.split(iline)
            if len(ifields) != 3:
                continue
            ilemma1, ilemma2, iwght = ifields
            if _check_word(ilemma1) and _check_word(ilemma2):
                ising.add_edge(normalize(ilemma1),
                               normalize(ilemma2), float(iwght),
                               a_add_missing=True)


def takamura(a_germanet, a_N, a_cc_file, a_pos, a_neg, a_neut, a_plot=None):
    """
    Method for extending sentiment lexicons using Takamura method

    @param a_germanet - GermaNet instance
    @param a_N - number of terms to extract
    @param a_cc_file - file containing coordinatively conjoined phrases
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded
    @param a_plot - name of file in which generated statics plots should be
                    saved (None if no plot should be generated)

    @return \c 0 on success, non-\c 0 otherwise
    """
    # estimate the number of terms to extract
    seed_set = a_pos | a_neg
    # create initial empty network
    ising = Ising()
    # populate network from GermaNet
    print("Adding GermaNet synsets...", end="", file=sys.stderr)
    _tkm_add_germanet(ising, a_germanet)
    print("done (Ising model has {:d} nodes)".format(ising.n_nodes),
          file=sys.stderr)
    # populate network from corpus
    print("Adding coordinate phrases from corpus...",
          end="", file=sys.stderr)
    _tkm_add_corpus(ising, a_cc_file)
    print("done (Ising model has {:d} nodes)".format(ising.n_nodes),
          file=sys.stderr)
    # reweight edges
    ising.reweight()
    # set fixed weights for words pertaining to the positive, negative, and
    # neutral set
    for ipos in a_pos:
        if ipos in ising:
            ising[ipos][FXD_WGHT_IDX] = 1.
        else:
            ising.add_node(ipos, 1.)
        ising[ipos][HAS_FXD_WGHT] = 1
    for ineg in a_neg:
        if ineg in ising:
            ising[ineg][FXD_WGHT_IDX] = -1.
        else:
            ising.add_node(ineg, -1.)
        ising[ineg][HAS_FXD_WGHT] = 1
    for ineut in a_neut:
        if ineut in ising:
            ising[ineut][FXD_WGHT_IDX] = 0.
        else:
            ising.add_node(ineut, 0.)
        ising[ineut][HAS_FXD_WGHT] = 1
    ising.train(a_plot=a_plot)
    # nodes = [inode[ITEM_IDX]
    # for inode in sorted(ising.nodes, key = lambda x: x[WGHT_IDX])
    #              if inode[ITEM_IDX] not in seed_set]
    seed_set |= a_neut
    nodes = [inode
             for inode in sorted(ising.nodes,
                                 key=lambda x: abs(x[WGHT_IDX]), reverse=True)
             if inode[ITEM_IDX] not in seed_set]
    seed_set.clear()
    # populate polarity sets and flush all terms to an external file
    i = 0
    with open(os.path.join("data", "ising_full.txt"), 'w') as ofile:
        for inode in nodes:
            # print("inode =", repr(inode), file = sys.stderr)
            # print("type(inode[WGHT_IDX]) =", type(inode[WGHT_IDX]),
            # file=sys.stderr)
            if isnan(inode[WGHT_IDX]):
                print(inode[ITEM_IDX].encode(ENCODING), "\t", inode[WGHT_IDX],
                      file=ofile)
            else:
                if i < a_N:
                    if inode[WGHT_IDX] > 0:
                        a_pos.add(inode[ITEM_IDX])
                    elif inode[WGHT_IDX] < 0:
                        a_neg.add(inode[ITEM_IDX])
                    else:
                        i -= 1
                    i += 1
                print(inode[ITEM_IDX].encode(ENCODING),
                      "\t{:f}".format(inode[WGHT_IDX]), file=ofile)


def main(a_argv):
    """Main method for generating sentiment lexicons

    @param a_argv - command-line arguments

    @return \c 0 on success, non-\c 0 otherwise

    """
    argparser = argparse.ArgumentParser(
        description="Script for generating sentiment lexicons.")
    # add type-specific subparsers
    subparsers = argparser.add_subparsers(
        help="lexicon expansion method to use", dest="dmethod"
    )

    subparser_awdallah = subparsers.add_parser(AWDALLAH,
                                               help="Awdallah's model"
                                               " (Awdallah et al., 2011)")
    subparser_awdallah.add_argument("--ext-syn-rels",
                                    help="use extended set of synonymous"
                                    " relations",
                                    action="store_true")
    subparser_awdallah.add_argument("--teleport",
                                    help="probability of a random"
                                    " teleport transition",
                                    type=float, default=0.)
    _add_cmn_opts(subparser_awdallah)

    subparser_bg = subparsers.add_parser(BG,
                                         help="Blair-Goldensohn's model"
                                         " (Blair-Goldensohn et al., 2008)")
    subparser_bg.add_argument("--ext-syn-rels",
                              help="use extended set of synonymous"
                              " relations",
                              action="store_true")
    _add_cmn_opts(subparser_bg)

    subparser_hu = subparsers.add_parser(HU,
                                         help="Hu/Liu model"
                                         " (Hu and Liu, 2004)")
    subparser_hu.add_argument("--ext-syn-rels",
                              help="use extended set of synonymous relations",
                              action="store_true")
    _add_cmn_opts(subparser_hu)

    subparser_esuli = subparsers.add_parser(ESULI,
                                            help="SentiWordNet model"
                                            " (Esuli and Sebastiani, 2005)")
    _add_cmn_opts(subparser_esuli)

    subparser_takamura = subparsers.add_parser(TAKAMURA,
                                               help="Ising spin model"
                                               " (Takamura, 2005)")
    subparser_takamura.add_argument("--plot", "-p",
                                    help="suffix of files in"
                                    " which to store the plot image",
                                    type=str, default="")
    _add_cmn_opts(subparser_takamura)
    subparser_takamura.add_argument(CC_FILE,
                                    help="file containing coordinatively"
                                    "conjoined phrases")
    subparser_takamura.add_argument("N",
                                    help="final number of additional"
                                    " terms to extract", type=int)
    args = argparser.parse_args(a_argv)

    # initialize GermaNet, if needed
    igermanet = None
    if GNET_DIR in args:
        print("Reading GermaNet synsets... ",
              end="", file=sys.stderr)
        igermanet = Germanet(getattr(args, GNET_DIR))
        if "form2lemma" in args and args.form2lemma is not None:
            global lemmatize
            lemmatize = _lemmatize
            _get_form2lemma(args.form2lemma)
        print("done", file=sys.stderr)

    # obtain lists of conjoined terms, if needed

    # read initial seed set
    print("Reading seed sets... ", end="", file=sys.stderr)
    _read_set(args.seed_set)
    print("done", file=sys.stderr)

    # only perform expansion if the number of seed terms is less than
    # the request number of polar items
        # apply requested method
    print("Expanding polarity sets... ", file=sys.stderr)
    if args.dmethod == AWDALLAH:
        new_terms = awdallah(igermanet, POS_SET, NEG_SET, NEUT_SET,
                             args.seed_pos, args.ext_syn_rels, args.teleport)
    elif args.dmethod == BG:
        new_terms = blair_goldensohn(igermanet, POS_SET, NEG_SET, NEUT_SET,
                                     args.seed_pos, args.ext_syn_rels)
    elif args.dmethod == ESULI:
        new_terms = esuli_sebastiani(igermanet, POS_SET, NEG_SET, NEUT_SET,
                                     args.seed_pos)
    elif args.dmethod == HU:
        new_terms = hu_liu(igermanet, POS_SET, NEG_SET, NEUT_SET,
                           args.seed_pos, args.ext_syn_rels)
    elif args.dmethod == TAKAMURA:
        N = args.N - (len(POS_SET) + len(NEG_SET))
        if N > 1:
            new_terms = takamura(igermanet, N, getattr(args, CC_FILE),
                                 POS_SET, NEG_SET, NEUT_SET,
                                 a_plot=args.plot or None)
    else:
        raise NotImplementedError
    print("Expanding polarity sets... done", file=sys.stderr)

    for iterm, itag, _ in new_terms:
        print("{:s}\t{:s}".format(iterm, itag).encode(ENCODING))

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
