#!/usr/bin/env python2.7
# -*- coding: utf-8; mode: python; -*-

"""
Module for reading and processing GemaNet files.

Constants:
POS - list of parts-of-speech present in GermaNet
RELTYPES - types of GermaNet relations

Classes:
Germanet - main class for processing GermaNet files

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from itertools import chain
from collections import defaultdict

import argparse
import codecs
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

##################################################################
# Variables and Constants
SKIP_RE = re.compile(r"\s+[1-9]")
ENCODING = "utf-8"
POS = [".adj", ".adv", ".noun", ".verb"]
RELSYM2NAME = {
    "~": "Hyponym",
    "~i": "Instance Hyponym",
    "!": "Antonym",
    "#m": "Member holonym",
    "#p": "Part holonym",
    "#s": "Substance holonym",
    "$": "Verb Group",
    "%m": "Member meronym",
    "%p": "Part meronym",
    "%s": "Substance meronym",
    "&": "Similar to",
    "*": "Entailment",
    "+": "Derivationally related form",
    "-c": "Member of this domain - TOPIC",
    "-r": "Member of this domain - REGION",
    "-u": "Member of this domain - USAGE",
    ";c": "Domain of synset - TOPIC",
    ";r": "Domain of synset - REGION",
    ";u": "Domain of synset - USAGE",
    "<": "Participle of verb",
    "=": "Attribute",
    ">": "Cause",
    "@": "Hypernym",
    "@i": "Instance Hypernym",
    "\\": "Derived from adjective",
    "^": "Also see"
}


##################################################################
# Class
class Wordnet(object):
    """
    Class for reading and pocessing GermaNet files

    Instance variables:
    lexid2lex - mapping from lexeme IDs to lexemes
    lex2lexid - mapping from lexemes to lexeme IDs
    lexid2synids - mapping from lexeme IDs to synset IDs
    synid2lexids - mapping from synset IDs to lexemes
    synid2defexmp - mapping from synset IDs to synset definitions and examples
    con_relations - adjacency lists of relations between synsets
    lex_relations - adjacency lists of relations between lexemes

    """

    def __init__(self, a_dir=os.getcwd()):
        """Class constructor.

        @param a_dir - directory containing GermaNet files

        """
        if not os.path.isdir(a_dir) or not os.access(a_dir, os.R_OK):
            raise RuntimeError("Can't read from directory: {:s}".format(a_dir))
        ## mapping from synset IDs to synset definitions and examples
        self.synid2defexmp = dict()
        ## mapping from synset IDs to part-of-speech categories
        self.synid2pos = dict()
        ## mapping from synset IDs to lexemes
        self.synid2lexemes = defaultdict(set)
        ## mapping from lexeme IDs to lexemes
        self.lexeme2synids = defaultdict(set)
        ## adjacency lists of relations between synsets
        self.relations = defaultdict(set)
        # parse synsets
        for ifile in chain.from_iterable(
                glob.iglob(os.path.join(a_dir, "data" + ipos))
                for ipos in POS):
            self._parse_synsets(ifile)
        assert self.lexeme2synids, \
            "No synset files found in directory {:s}".format(a_dir)

    def _parse_synsets(self, a_fname):
        """Parse GemaNet XML file

        @param a_fname - name of input file

        @return \c void

        """
        ptr_sym = ""
        i = w_cnt = rel_cnt = 0
        ilex = toks = syn_id = pos = trg_id = trg_synid = trg_pos = None
        with codecs.open(a_fname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.rstrip()
                if SKIP_RE.match(iline):
                    continue
                # print("iline = ", repr(iline), file=sys.stderr)
                toks = iline.split()
                syn_id, pos = toks[0], toks[2]
                syn_id = (syn_id, pos)
                self.synid2pos[syn_id] = pos
                # print("syn_id =", repr(syn_id), file=sys.stderr)
                # print("pos =", repr(pos), file=sys.stderr)
                w_cnt = int(toks[3], 16)
                # print("w_cnt =", repr(w_cnt), file=sys.stderr)
                # read lexemes
                for j in xrange(4, 4 + w_cnt * 2, 2):
                    ilex = toks[j]
                    self.synid2lexemes[syn_id].add(ilex)
                    self.lexeme2synids[ilex].add(syn_id)
                # print("self.synid2lexemes[syn_id] =",
                #       repr(self.synid2lexemes[syn_id]), file=sys.stderr)
                # print("self.lexeme2synids[ilex] =",
                #       repr(self.lexeme2synids[ilex]), file=sys.stderr)
                # read relations
                i = 4 + w_cnt * 2
                rel_cnt = int(toks[i])
                i += 1
                # print("rel_cnt =",
                #       repr(rel_cnt), file=sys.stderr)
                # print("i =", repr(i), file=sys.stderr)
                for j in xrange(i, i + rel_cnt * 4, 4):
                    ptr_sym, trg_synid, trg_pos, _ = toks[j:j+4]
                    # print("ptr_sym =",
                    #       repr(ptr_sym), file=sys.stderr)
                    # print("trg_synid =",
                    #       repr(trg_synid), file=sys.stderr)
                    # print("trg_pos =",
                    #       repr(trg_pos), file=sys.stderr)
                    trg_id = (trg_synid, trg_pos)
                    self.relations[syn_id].add((trg_id, RELSYM2NAME[ptr_sym]))
                i += rel_cnt * 4
                # print("i =", repr(i), file=sys.stderr)
                if pos == 'v':
                    f_cnt = int(toks[i])
                    i += f_cnt * 3 + 1
                assert toks[i] == '|', \
                    "Invalid line format '{:s}' token {:d} expected" \
                    " to be '|', but it is '{:s}' ".format(repr(iline), i,
                                                           repr(toks[i]))
                self.synid2defexmp[syn_id] = ' '.join(toks[i + 1:])
