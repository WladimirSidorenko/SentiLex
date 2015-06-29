#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""
Script for extracting coordinatively conjoined phrases from DG trees

USAGE:
extract_cc.py [OPTIONS] [INPUT_FILEs]
"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function
from conll import CONLLWord, CONLLSentence
from germanet import normalize

from itertools import combinations

import argparse
import codecs
import re
import string
import sys

##################################################################
# Variables and Constants
VERBOSE = False
PUNCT_RE = re.compile(r"^(?:" + '|'.join([re.escape(c) for c in string.punctuation]) + \
                          ")+$")
ESC_CHAR = ""
ENCODING = "utf-8"
CC_RELATIONS = set(["CD", "CJ"])
ADVERS_CC = set([normalize(w) for w in ["aber"]])
COORD_CC = set([normalize(w) for w in ["und", "oder", ","]])

##################################################################
# Methods
def _find_roots(a_tree):
    """
    Find roots of DG tree

    @param a_tree - DG tree to process

    @return list of root indices
    """
    for inode in a_tree:
        if inode.phead == '0':
            yield inode

def _process_cc_helper(a_tree, a_iroot, a_cc_main, a_cc_seen = False):
    """
    Find coordinarively conjoined phrases in DG tree

    @param a_tree - DG tree to process
    @param a_index - root node of the tree
    @param a_cc_main - main list of coordinarively conjoined phrases to be
                       populated
    @param a_cc_seen - particular list of coordinarively conjoined phrases
                       coming from parent

    @return \c void
    """
    if a_tree.is_empty():
        return

    ret = [normalize(a_iroot.plemma)]
    # print("ret =", repr(ret), file = sys.stderr)
    # extract coordinarively conjoined chains
    for ichild in a_tree.children[a_iroot.idx]:
        # print("ichild.pdeprel =", repr(ichild.pdeprel), file = sys.stderr)
        if ichild.pdeprel in CC_RELATIONS:
            ret.append(_process_cc_helper(a_tree, ichild, a_cc_main, True))
        else:
            a_cc_main += _process_cc_helper(a_tree, ichild, a_cc_main, False)
    if len(ret) != 1:
        if a_cc_seen:
            return ret
        else:
            a_cc_main.append(ret)
            return []
    elif a_cc_seen:
        return ret[0]
    else:
        return []

def _output_cc_helper(a_cc_list):
    """
    Extract sets of coordinatively and adversatively conjoined terms

    @param a_cc_list - list of conjoined phrases

    @return 2-tuple with sets of coordinatively and adversatively conjoined terms
    """
    coord = set(); advers = set()
    ret = (coord, advers)
    trg_coord = trg_advers = chld_coord = chld_advers = None

    chld_ret = None
    if type(a_cc_list) == list:
        if not a_cc_list:
            return ret
        a_cc_list[0] = normalize(a_cc_list[0])
        if a_cc_list[0] in ADVERS_CC:
            trg_coord, trg_advers = advers, coord
        else:
            if a_cc_list[0] not in COORD_CC:
                if VERBOSE:
                    print("WARNING: Unknown coordinative conjunction: '{:s}'".format(repr(a_cc_list[0])), \
                              file = sys.stderr)
                coord.add(a_cc_list[0])
            trg_coord, trg_advers = coord, advers
        for chld in a_cc_list[1:]:
            chld_coord, chld_advers = _output_cc_helper(chld)
            trg_coord |= chld_coord; trg_advers |= chld_advers
    else:
        coord.add(a_cc_list)
    return ret

def _output_cc(a_cc_list):
    """
    Output coordinatively conjoined phrases

    @param a_cc_list - list of coordinatively conjoined phrases

    @return \c void
    """
    coord = advers = None
    for cc in a_cc_list:
        coord, advers = _output_cc_helper(cc)
        coord = set(w for w in coord if not PUNCT_RE.match(w))
        advers = set(w for w in advers if not PUNCT_RE.match(w))
        # output all possible pairs of coordinatively conjoined phrases
        for w1, w2 in combinations(coord, 2):
            print("{:s}\t{:s}\t1".format(w1, w2).encode(ENCODING))
        for w1, w2 in combinations(advers, 2):
            print("{:s}\t{:s}\t1".format(w1, w2).encode(ENCODING))
        # output all possible pairs of adversatively conjoined phrases
        for w1 in coord:
            for w2 in advers:
                print("{:s}\t{:s}\t-1".format(w1, w2).encode(ENCODING))

def process_cc(a_tree):
    """
    Output all coordinarively conjoined phrases found in DG tree

    @param a_tree - DG tree to process

    @return \c void
    """
    if a_tree.is_empty():
        return
    cc_list = []
    # find trees roots
    for iroot in _find_roots(a_tree):
        _process_cc_helper(a_tree, iroot, cc_list, False)
        # extract coordinarively conjoined chains
        # print(repr(cc_list), file = sys.stderr)
        _output_cc(cc_list)
        del cc_list[:]
    # clear sentence
    a_tree.clear()

def parse_file(a_fname):
    """
    Output all coordinarively conjoined phrases from DG trees

    @param a_fname - name of the input file

    @return \c void
    """
    try:
        # open file for reading
        ifile = sys.stdin
        if a_fname != "-":
            ifile = codecs.open(a_fname, 'r', ENCODING)
        # iterate over file's lines
        conll_sentence = CONLLSentence()
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                process_cc(conll_sentence)
            elif iline[0] == ESC_CHAR:
                process_cc(conll_sentence)
                continue
            else:
                conll_sentence.push_word(CONLLWord(iline))
        process_cc(conll_sentence)
    finally:
        ifile.close()

def main(a_argv):
    """
    Main method for extracting coordinatively conjoined phrases from DG trees

    @param a_argv - command-line arguments

    @return \c 0 on success, non-\c 0 otherwise
    """
    argparser = argparse.ArgumentParser(description = """Script for extracting\
 coordinatively conjoined phrases from DG trees""")
    argparser.add_argument("files", help = "input files containing CONLL DG trees", \
                               nargs = '*')
    args = argparser.parse_args(a_argv)

    if not args.files:
        args.files.append("-")

    for a_fname in args.files:
        parse_file(a_fname)

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
