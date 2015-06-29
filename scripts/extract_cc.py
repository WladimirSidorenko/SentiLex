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
import sys

##################################################################
# Variables and Constants
ESC_CHAR = ""
ENCODING = "utf-8"
CC_RELATIONS = set(["CD", "CJ"])
ADVERS_CC = set([normalize(w) for w in ("aber")])
COORD_CC = set([normalize(w) for w in ("und", "oder")])

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
    print(repr(cc_list), file = sys.stderr)
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
