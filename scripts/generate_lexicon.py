#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Script for generating sentiment lexica.

USAGE:
generate_lexicon.py [OPTIONS] [INPUT_FILES]

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import POSITIVE, NEGATIVE, NEUTRAL, STOP_WORDS, \
    FORM2LEMMA, INFORMATIVE_TAGS, TAB_RE, NONMATCH_RE, ENCODING
from germanet import Germanet, normalize, POS

from awdallah import awdallah
from blair_goldensohn import blair_goldensohn
from esuli_sebastiani import esuli_sebastiani
from hu_liu import hu_liu
from kim_hovy import kim_hovy
from kiritchenko import kiritchenko
from rao import rao_min_cut, rao_lbl_prop
from severyn import severyn
from takamura import takamura
from velikovich import velikovich, DFLT_T

import argparse
import codecs
import os
import re
import string
import sys

##################################################################
# Imports
COMMENT = "###"
CORPUS_FILES = "corpus_files"
GNET_DIR = "germanet_dir"
CC_FILE = "cc_file"

VERBOSE = False

AWDALLAH = "awdallah"
BG = "blair-goldensohn"
ESULI = "esuli-sebastiani"
HU = "hu-liu"
KIM = "kim-hovy"
KIRITCHENKO = "kiritchenko"
RAO_LBL_PROP = "rao-lbl-prop"
RAO_MIN_CUT = "rao-min-cut"
SEVERYN = "severyn"
TAKAMURA = "takamura"
VELIKOVICH = "velikovich"

W_DELIM_RE = re.compile(r'(?:\s|{:s})+'.format(
    '|'.join([re.escape(c) for c in string.punctuation])))
POS_SET = set()                 # set of positive terms
NEG_SET = set()                 # set of negative terms
NEUT_SET = set()                # set of neutral terms
POS_RE = None
NEG_RE = None

REGEXP = "REGEXP"
SEED_RE_SUPPORTED_METHODS = set([KIRITCHENKO, SEVERYN, TAKAMURA, VELIKOVICH])


##################################################################
# Main
def normalize_reg(a_reg):
    """Wrap string representing regular expression in brackets.

    @param a_reg - string representing regular expression
    @type unicode

    @return regexp string wrapped in parentheses
    @type unicode

    """
    return "(?:" + a_reg + ")"


def join_regs(a_regs):
    """Compile list of strings into a single regular expression.

    @param a_regs - string representing regular expression
    @type list

    @return regexp string wrapped in parentheses
    @type unicode

    """
    return re.compile(r"(?:^|\s)(?:" + '|'.join(a_regs) + r")(?:$|\s)")


def _add_cmn_opts(a_parser, a_add_ext_opts=True):
    """Add options common to all option parsers.

    @param a_parser - argument parser to add options to
    @param a_add_ext_opts - add an option for extended synonym relations

    @return \c void

    """
    a_parser.add_argument("--seed-pos",
                          help="part-of-speech of seed synsets"
                          " ('none' for no restriction)",
                          choices=["none"] + [p[:-1] for p in POS],
                          default="none"
                         )
    a_parser.add_argument("--form2lemma", "-l",
                          help="file containing Germanet form-lemma"
                          " correspondences", type=str)
    a_parser.add_argument("seed_set",
                          help="initial seed set of positive,"
                          " negative, and neutral terms")
    a_parser.add_argument(GNET_DIR,
                          help="directory containing GermaNet files")
    if a_add_ext_opts:
        a_parser.add_argument("--ext-syn-rels",
                              help="use extended set of synonymous"
                              " relations", action="store_true")


def _get_dflt_lexicon(a_pos, a_neg):
    """Generate default lexicon by putting in it terms from seed set.

    @param a_pos - set of positive terms
    @param a_neg - set of negative terms

    @return list(3-tuple) - list of seed set terms with uniform scores and
      polarities

    """
    return [(w, POSITIVE, 1.) for w in a_pos] \
        + [(w, NEGATIVE, -1.) for w in a_neg]


def _get_form2lemma(a_fname):
    """Read file containing form/lemma correspodences

    @param a_fname - name of input file

    @return void (correspondences are read into global variables)

    """
    global STOP_WORDS, FORM2LEMMA

    if not os.path.isfile(a_fname) or not os.access(a_fname, os.R_OK):
        raise RuntimeError("Cannot read from file '{:s}'".format(
            a_fname))

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

    @return void

    """
    global POS_SET, NEG_SET, NEUT_SET, POS_RE, NEG_RE
    fields = []
    pos_regs = []
    neg_regs = []
    with codecs.open(a_fname, 'r',
                     encoding=ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            elif iline.startswith(COMMENT):
                # maybe, we will later introduce some special comments
                continue
            fields = TAB_RE.split(iline)
            if len(fields) > 2 and fields[2] == REGEXP:
                if fields[1] == POSITIVE:
                    pos_regs.append(normalize_reg(fields[0]))
                elif fields[1] == NEGATIVE:
                    neg_regs.append(normalize_reg(fields[0]))
                else:
                    raise NotImplementedError(
                        "Regular expressions are not supported"
                        " for non-polar classes.")
                continue
            if fields[1] == POSITIVE:
                POS_SET.add(normalize(fields[0]))
            elif fields[1] == NEGATIVE:
                NEG_SET.add(normalize(fields[0]))
            elif fields[1] == NEUTRAL:
                NEUT_SET.add(normalize(fields[0]))
            else:
                raise RuntimeError(
                    "Unknown field specification: {:s}".format(fields[-1]))
    if pos_regs:
        POS_RE = join_regs(pos_regs)
    if neg_regs:
        NEG_RE = join_regs(neg_regs)


def main(a_argv):
    """Main method for generating sentiment lexicons

    @param a_argv - command-line arguments

    @return \c 0 on success, non-\c 0 otherwise

    """
    global POS_RE, NEG_RE

    argparser = argparse.ArgumentParser(
        description="Script for generating sentiment lexicons.")
    # add type-specific subparsers
    subparsers = argparser.add_subparsers(
        help="lexicon expansion method to use", dest="dmethod"
    )

    subparser_awdallah = subparsers.add_parser(AWDALLAH,
                                               help="Awdallah's model"
                                               " (Awdallah et al., 2011)")
    subparser_awdallah.add_argument("--teleport",
                                    help="probability of a random"
                                    " teleport transition",
                                    type=float, default=0.)
    _add_cmn_opts(subparser_awdallah)

    subparser_bg = subparsers.add_parser(BG,
                                         help="Blair-Goldensohn's model"
                                         " (Blair-Goldensohn et al., 2008)")
    _add_cmn_opts(subparser_bg)

    subparser_hu = subparsers.add_parser(HU,
                                         help="Hu/Liu model"
                                         " (Hu and Liu, 2004)")
    _add_cmn_opts(subparser_hu)

    subparser_esuli = subparsers.add_parser(ESULI,
                                            help="SentiWordNet model"
                                            " (Esuli and Sebastiani, 2005)")
    _add_cmn_opts(subparser_esuli)

    subparser_kim = subparsers.add_parser(
        KIM, help="Kim's method (Kim and Hovy, 2004)")
    _add_cmn_opts(subparser_kim)

    subparser_kiritchenko = subparsers.add_parser(
        KIRITCHENKO, help="Kiritchenko's method (Kiritchenko et al., 2014)")
    subparser_kiritchenko.add_argument("seed_set",
                                       help="initial seed set of positive,"
                                       " negative, and neutral terms")
    subparser_kiritchenko.add_argument("N",
                                       help="final number of additional"
                                       " terms to extract", type=int)
    subparser_kiritchenko.add_argument(CORPUS_FILES, nargs='+',
                                       help="tagged and lemmatzied corpus"
                                       " files")

    subparser_rao_min_cut = subparsers.add_parser(
        RAO_MIN_CUT, help="Rao/Ravichandran's min-cut model"
        " (Rao and Ravichandran, 2009)")
    _add_cmn_opts(subparser_rao_min_cut)

    subparser_rao_lbl_prop = subparsers.add_parser(
        RAO_LBL_PROP, help="Rao/Ravichandran's label propagation model"
        " (Rao and Ravichandran, 2009)")
    _add_cmn_opts(subparser_rao_lbl_prop)

    subparser_severyn = subparsers.add_parser(
        SEVERYN, help="Severyn's method (Severyn and Moschitti, 2014)")
    subparser_severyn.add_argument("seed_set",
                                   help="initial seed set of positive,"
                                   " negative, and neutral terms")
    subparser_severyn.add_argument("N",
                                   help="final number of additional"
                                   " terms to extract", type=int)
    subparser_severyn.add_argument(CORPUS_FILES, nargs='+',
                                   help="tagged and lemmatzied corpus"
                                   " files")

    subparser_takamura = subparsers.add_parser(TAKAMURA,
                                               help="Ising spin model"
                                               " (Takamura, 2005)")
    subparser_takamura.add_argument("--plot", "-p",
                                    help="suffix of files in"
                                    " which to store the plot image",
                                    type=str, default="")
    _add_cmn_opts(subparser_takamura, False)
    subparser_takamura.add_argument(CC_FILE,
                                    help="file containing coordinatively"
                                    "conjoined phrases")
    subparser_takamura.add_argument("N",
                                    help="final number of additional"
                                    " terms to extract", type=int)
    subparser_velikovich = subparsers.add_parser(VELIKOVICH,
                                                 help="Velikovich's model"
                                                 " (Velikovich et al., 2010)")
    subparser_velikovich.add_argument("-t",
                                      help="maximum number of iterations",
                                      type=int, default=DFLT_T)
    subparser_velikovich.add_argument("seed_set",
                                      help="initial seed set of positive,"
                                      " negative, and neutral terms")
    subparser_velikovich.add_argument("N",
                                      help="final number of additional"
                                      " terms to extract", type=int)
    subparser_velikovich.add_argument(CORPUS_FILES, nargs='+',
                                      help="tagged lemmatized files of the"
                                      " original corpus")
    args = argparser.parse_args(a_argv)

    # initialize GermaNet, if needed
    igermanet = None
    if GNET_DIR in args:
        print("Reading GermaNet synsets... ",
              end="", file=sys.stderr)
        igermanet = Germanet(getattr(args, GNET_DIR))
        print("done", file=sys.stderr)
    # For Takamura's method, `form2lemma' will contain lemmas of words
    # appearing in Germanet glosses.  For corpus based approaches, it will
    # contain lemmas of words contained in the corpus.
    if "form2lemma" in args and args.form2lemma is not None:
        _get_form2lemma(args.form2lemma)

    # obtain lists of conjoined terms, if needed

    # read initial seed set
    print("Reading seed sets... ", end="", file=sys.stderr)
    _read_set(args.seed_set)
    print("done", file=sys.stderr)

    # only perform expansion if the number of seed terms is less than
    # the requested number of polar items
        # apply requested method
    print("Expanding polarity sets... ", file=sys.stderr)
    if "seed_pos" in args and args.seed_pos \
       and args.seed_pos.lower() == "none":
        args.seed_pos = None

    # check whether seed sets specified any regular expressions and whether
    # these are supported by the respective methods
    if args.dmethod not in SEED_RE_SUPPORTED_METHODS:
        if POS_RE is not None or NEG_RE is not None:
            raise NotImplementedError("Method {:s} does not support"
                                      " regular expressions in seed sets.")
    else:
        # set missing regular expressions to the never-matching ones
        if POS_RE is None:
            POS_RE = NONMATCH_RE
        if NEG_RE is None:
            NEG_RE = NONMATCH_RE
    # run the actual algorithms
    if args.dmethod == AWDALLAH:
        new_terms = awdallah(igermanet, POS_SET, NEG_SET, NEUT_SET,
                             args.seed_pos, args.ext_syn_rels, args.teleport)
    elif args.dmethod == BG:
        new_terms = blair_goldensohn(igermanet, POS_SET, NEG_SET, NEUT_SET,
                                     args.seed_pos, args.ext_syn_rels)
    elif args.dmethod == ESULI:
        new_terms = esuli_sebastiani(igermanet, POS_SET, NEG_SET, NEUT_SET,
                                     args.seed_pos, args.ext_syn_rels)
    elif args.dmethod == HU:
        new_terms = hu_liu(igermanet, POS_SET, NEG_SET, NEUT_SET,
                           args.seed_pos, args.ext_syn_rels)
    elif args.dmethod == KIM:
        new_terms = kim_hovy(igermanet, POS_SET, NEG_SET, NEUT_SET,
                             args.seed_pos, args.ext_syn_rels)
    elif args.dmethod == KIRITCHENKO:
        N = args.N - (len(POS_SET) + len(NEG_SET))
        if N == 0:
            new_terms = _get_dflt_lexicon(POS_SET, NEG_SET)
        else:
            new_terms = kiritchenko(N, getattr(args, CORPUS_FILES),
                                    POS_SET, NEG_SET, POS_RE, NEG_RE)
    elif args.dmethod == RAO_MIN_CUT:
        new_terms = rao_min_cut(igermanet, POS_SET, NEG_SET, NEUT_SET,
                                args.seed_pos, args.ext_syn_rels)
    elif args.dmethod == RAO_LBL_PROP:
        new_terms = rao_lbl_prop(igermanet, POS_SET, NEG_SET, NEUT_SET,
                                 args.seed_pos, args.ext_syn_rels)
    elif args.dmethod == SEVERYN:
        N = args.N - (len(POS_SET) + len(NEG_SET))
        if N == 0:
            new_terms = _get_dflt_lexicon(POS_SET, NEG_SET)
        else:
            new_terms = severyn(N, getattr(args, CORPUS_FILES),
                                POS_SET, NEG_SET, POS_RE, NEG_RE)
    elif args.dmethod == TAKAMURA:
        N = args.N - (len(POS_SET) + len(NEG_SET))
        if N == 0:
            new_terms = _get_dflt_lexicon(POS_SET, NEG_SET)
        else:
            new_terms = takamura(igermanet, N, getattr(args, CC_FILE),
                                 POS_SET, NEG_SET, NEUT_SET,
                                 a_plot=args.plot or None,
                                 a_pos_re=POS_RE, a_neg_re=NEG_RE)
    elif args.dmethod == VELIKOVICH:
        N = args.N - (len(POS_SET) + len(NEG_SET))
        if N == 0:
            new_terms = _get_dflt_lexicon(POS_SET, NEG_SET)
        else:
            new_terms = velikovich(N, args.t, getattr(args, CORPUS_FILES),
                                   POS_SET, NEG_SET, POS_RE, NEG_RE)
    else:
        raise NotImplementedError
    print("Expanding polarity sets... done", file=sys.stderr)

    for iterm, itag, iscore in new_terms:
        print("{:s}\t{:s}\t{:f}".format(iterm, itag, iscore).encode(ENCODING))

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
