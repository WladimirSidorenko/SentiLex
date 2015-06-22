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
RELTYPES = ["con_rel", "lex_rel"]
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
    return SZET_RE.sub("ss", SPACE_RE.sub(' ', a_string.lower()))

##################################################################
# Class
class Germanet(object):
    """
    Class for reading and pocessing GermaNet files

    Instance variables:
    lex2synids - mapping from lexeme IDs to synset IDs
    synid2lex - mapping from synset IDs to lexemes
    synid2defexmp - mapping from synset IDs to synset definitions and examples
    relations - adjacency list of relations
    """

    def __init__(self, a_dir = os.getcwd()):
        """
        Class constructor

        @param a_dir - directory containing GermaNet files
        """
        if not os.path.isdir(a_dir) or not os.access(a_dir, os.R_OK):
            raise RuntimeError("Can't read from directory: {:s}".format(a_dir))
        self.synid2defexmp = dict()
        self.lex2synids = defaultdict(set)
        self.synid2lex = defaultdict(set)
        self.relations = defaultdict(set)
        # parse synsets
        for ifile in chain.from_iterable(glob.iglob(os.path.join(a_dir, ipos + '*')) \
                                             for ipos in POS):
            self._parse_synsets(ifile)

        assert self.lex2synids, "No synset files found in directory {:s}".format(a_dir)
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
                for iform in ilex.iterfind("./orthForm"):
                    lex = normalize(iform.text)
                    self.lex2synids[lex].add(synid)
                    self.synid2lex[synid].add(lex)

    def _parse_relations(self, a_fname):
        """
        Parse GemaNet relations file

        @param a_fname - name of input file

        @return \c void
        """
        ifrom = ito = iinverse = None
        idoc = ET.parse(a_fname).getroot()
        for rel_type in RELTYPES:
            for irel in idoc.iterfind(rel_type):
                ifrom, ito = irel.get("from"), irel.get("to")
                self.relations[ifrom].add((ito, irel.get("name")))
                iinverse = irel.get("inv")
                if iinverse:
                    self.relations[ito].add((ifrom, iinverse))
