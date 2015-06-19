#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
Script for merging polarity lexicons

USAGE:
merge_lexicons.py gpc_dir sws_dir zrch_dir

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from gpc import GPC, NEGATIVE, POSITIVE, NEUTRAL
from sws import SWS
from zrch import ZRCH

import argparse
import sys

##################################################################
# Constants and Variables
INTERSECT = "intersect"
UNION = "union"
ENCODING = "utf-8"
DELIM = '\t'

##################################################################
# Methods
def _extend_set(a_set, a_union, a_form2lemma1, a_form2lemma2):
    """
    Extend exisiting set by adding to it forms whose lemmas are in the set

    @param a_set - set to be expanded
    @param a_union - container of additional terms (should subsume `a_set`)
    @param a_form2lemma1 - dictionary mapping forms to lemmas
    @param a_form2lemma2 - dictionary mapping forms to lemmas

    @return pointer to the new extended set
    """
    return set(term for term in a_union if term in a_set or \
                          a_form2lemma1.get(term) in a_set or \
                    a_form2lemma2.get(term) in a_set)
def main():
    """
    Obtain union or intersection of entries in polar lexicons

    @return \c 0 on success, non-\c 0 otherwise
    """
    argparser = argparse.ArgumentParser(description = "Script for merging polarity lexicons")
    argparser.add_argument("--operation", \
                               help = "type of operation to perform on dictionary entries", \
                               type = str, choices = [UNION, INTERSECT], default = INTERSECT)
    # argparser.add_argument("gpc_dir", help = "directory containing German Polarity Clues")
    # argparser.add_argument("sws_dir", help = "directory containing SentiWS lexicon")
    argparser.add_argument("zrch_dir", help = "directory containing Zurich polarity lexicon")
    args = argparser.parse_args()

    # initialize dictionaries
    gpc = GPC(args.gpc_dir)
    sws = SWS(args.sws_dir)
    zrch = ZRCH(args.zrch_dir)

    # create union of all terms
    pos_union = set(gpc.positive.keys()) | set(sws.positive.keys()) | set(zrch.positive.keys())
    neg_union = set(gpc.negative.keys()) | set(sws.negative.keys()) | set(zrch.negative.keys())
    neut_union = set(gpc.neutral.keys()) | set(zrch.neutral.keys())

    if args.operation == INTERSECT:
        pos_set = set(gpc.positive.keys()) & set(sws.positive.keys()) & set(sws.positive.keys())
        pos_set = _extend_set(pos_set, pos_union, gpc.form2lemma, sws.form2lemma)
        pos_union.clear()

        neg_set = set(gpc.negative.keys()) & set(sws.negative.keys()) & set(zrch.negative.keys())
        neg_set = _extend_set(neg_set, neg_union, gpc.form2lemma, sws.form2lemma)
        neg_union.clear()

        neut_set = set(gpc.neutral.keys()) & set(zrch.neutral.keys())
        neut_set = _extend_set(neut_set, neut_union, gpc.form2lemma, sws.form2lemma)
        neut_union.clear()
    elif args.operation == UNION:
        pos_set = pos_union
        neg_set = neg_union
        neut_set = neut_union
    else:
        raise RuntimeError("Unrecognized operation type: '{:s}'".format(args.operation))

    for iset, iclass in ((pos_set, POSITIVE), (neg_set, POSITIVE), (neut_set, NEUTRAL)):
        for iword in sorted(iset):
            print((iword + DELIM + iclass).encode(ENCODING))

##################################################################
# Main
if __name__ == "__main__":
    main()
