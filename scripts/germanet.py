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
    """Class for reading and pocessing GermaNet files

    Instance variables:
      lexid2lex - mapping from lexeme IDs to lexemes
      lex2lexid - mapping from lexemes to lexeme IDs
      lexid2synids - mapping from lexeme IDs to synset IDs
      synid2lexids - mapping from synset IDs to lexemes
      synid2defexmp - mapping from synset IDs to synset definitions
        and examples
      con_relations - adjacency lists of relations between synsets
      lex_relations - adjacency lists of relations between lexemes

    """

    def __init__(self, a_dir=os.getcwd()):
        """
        Class constructor

        @param a_dir - directory containing GermaNet files
        """
        if not os.path.isdir(a_dir) or not os.access(a_dir, os.R_OK):
            raise RuntimeError("Can't read from directory: {:s}".format(a_dir))
        ## mapping from synset IDs to synset definitions and examples
        self.synid2defexmp = dict()
        ## mapping from synset IDs to part-of-speech categories
        self.synid2pos = dict()
        ## mapping from lexeme IDs to lexemes
        self.lexid2lex = defaultdict(set)
        ## mapping from lexeme IDs to lexemes
        self.lex2lexid = defaultdict(set)
        ## mapping from lexeme IDs to synset IDs
        self.lexid2synids = defaultdict(set)
        ## mapping from synset IDs to lexemes
        self.synid2lexids = defaultdict(set)
        ## adjacency lists of relations between synsets
        self.con_relations = defaultdict(set)
        ## adjacency lists of relations between lexemes
        self.lex_relations = defaultdict(set)
        # parse synsets
        for ifile in chain.from_iterable(
                glob.iglob(os.path.join(a_dir, ipos + '*'))
                for ipos in POS):
            self._parse_synsets(ifile)
        assert self.lexid2synids, \
            "No synset files found in directory {:s}".format(a_dir)
        # parse wiktionary paraphrases
        for ifile in \
            chain.from_iterable(
                glob.iglob(os.path.join(a_dir, "wiktionaryParaphrases-"
                                        + ipos[:-1] + ".xml"))
                for ipos in POS):
            self._parse_wiktionary(ifile)
        # parse relations
        self._parse_relations(os.path.join(a_dir, "gn_relations.xml"))

    def _parse_synsets(self, a_fname):
        """
        Parse GemaNet XML file

        @param a_fname - name of input file

        @return \c void
        """
        lexid = ""
        synid = ""
        lex = ""
        iparaphrase = ilex = iform = None
        idoc = ET.parse(a_fname).getroot()
        for isynset in idoc.iterfind("synset"):
            synid = isynset.get("id")
            self.synid2pos[synid] = isynset.get("category")
            iparaphrase = isynset.find("./paraphrase")
            assert synid not in self.synid2defexmp, \
                "Duplicate description of synset {:s}".format(syn_id)
            self.synid2defexmp[synid] = (["" if iparaphrase is None
                                          else iparaphrase.text],
                                         [el.text
                                          for el in
                                          isynset.iterfind(".//example/text")])
            for ilex in isynset.iterfind("./lexUnit"):
                lexid = ilex.get("id")
                self.lexid2synids[lexid].add(synid)
                self.synid2lexids[synid].add(lexid)
                for iform in ilex.iterfind("./orthForm"):
                    lex = normalize(iform.text)
                    self.lexid2lex[lexid].add(lex)
                    self.lex2lexid[lex].add(lexid)

    def _parse_wiktionary(self, a_fname):
        """Parse wiktionary file with synset definitions

        @param a_fname - name of the Wi
        """
        ilexid = idef = None
        itree = ET.parse(a_fname).getroot()
        for wpara in itree.iterfind("wiktionaryParaphrase"):
            ilexid = wpara.get("lexUnitId")
            idef = wpara.get("wiktionarySense")
            # add definition to relevant synsets
            for isynid in self.lexid2synids[ilexid]:
                self.synid2defexmp[isynid][0].append(idef)

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
