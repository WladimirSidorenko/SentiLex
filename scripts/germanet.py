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
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

##################################################################
# Variables and Constants
POS = ["adj.", "nomen.", "verben."]
CON_REL = "con_rel"
LEX_REL = "lex_rel"
RELTYPES = [CON_REL, LEX_REL]
SZET_RE = re.compile('ß', re.U)
SPACE_RE = re.compile('(?:\t| \s)+')

##################################################################
# Methods
def normalize(a_string):
    """
    Lowercase string and replace multiple whitespaces.

    @param a_string - string to normalize

    @return normalized string version
    """
    return SZET_RE.sub("ss", SPACE_RE.sub(' ', a_string.lower())).strip()

##################################################################
# Class
class Germanet(object):
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

    def __init__(self, a_dir = os.getcwd()):
        """
        Class constructor

        @param a_dir - directory containing GermaNet files
        """
        if not os.path.isdir(a_dir) or not os.access(a_dir, os.R_OK):
            raise RuntimeError("Can't read from directory: {:s}".format(a_dir))
        self.synid2defexmp = dict()
        self.lexid2lex = dict(); self.lex2lexid = dict()
        self.lexid2synids = defaultdict(set)
        self.synid2lexids = defaultdict(set)
        self.con_relations = defaultdict(set)
        self.lex_relations = defaultdict(set)
        # parse synsets
        for ifile in chain.from_iterable(glob.iglob(os.path.join(a_dir, ipos + '*')) \
                                             for ipos in POS):
            self._parse_synsets(ifile)

        assert self.lexid2synids, "No synset files found in directory {:s}".format(a_dir)
        # parse relations
        self._parse_relations(os.path.join(a_dir, "gn_relations.xml"))

    def _parse_synsets(self, a_fname):
        """
        Parse GemaNet XML file

        @param a_fname - name of input file

        @return \c void
        """
        lexid = ""; synid = ""; lex = ""
        iparaphrase = ilex = iform = None
        idoc = ET.parse(a_fname).getroot()
        for isynset in idoc.iterfind("synset"):
            synid = isynset.get("id")
            iparaphrase = isynset.find("./paraphrase")
            assert synid not in self.synid2defexmp, \
                "Duplicate description of synset {:s}".format(syn_id)
            self.synid2defexmp[synid] = (("" if iparaphrase is None else iparaphrase.text), \
                                             [el.text for el in isynset.iterfind(".//example/text")])
            for ilex in isynset.iterfind("./lexUnit"):
                lexid = ilex.get("id")
                self.lexid2synids[lexid].add(synid)
                self.synid2lexids[synid].add(lexid)
                for iform in ilex.iterfind("./orthForm"):
                    lex = normalize(iform.text)
                    if lexid in self.lexid2lex:
                        assert lex not in self.lex2lexid, "Multiple lexeme id's specified for lemma {:s}".format(lex)
                        self.lexid2lex[lexid].add(lex)
                    else:
                        self.lexid2lex[lexid] = set([lex])
                        self.lex2lexid[lex] = lexid

    def _parse_relations(self, a_fname):
        """
        Parse GemaNet relations file

        @param a_fname - name of input file

        @return \c void
        """
        ifrom = ito = iinverse = None
        idoc = ET.parse(a_fname).getroot()
        # populate con relations
        for irel in idoc.iterfind(CON_REL):
            ifrom, ito = irel.get("from"), irel.get("to")
            self.con_relations[ifrom].add((ito, irel.get("name")))
            iinverse = irel.get("inv")
            if iinverse:
                self.con_relations[ito].add((ifrom, iinverse))
        # populate lex relations
        for irel in idoc.iterfind(LEX_REL):
            ifrom, ito = irel.get("from"), irel.get("to")
            self.lex_relations[ifrom].add((ito, irel.get("name")))
            iinverse = irel.get("inv")
            if iinverse:
                self.lex_relations[ito].add((ifrom, iinverse))
