#!/usr/bin/env python

"""
Script for converting GemaNet XML to TSV format.

@param dir - directory containing GermaNet files (pwd by default)
"""

##################################################################
# Imports
from itertools import chain

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

##################################################################
# Variables and Constants
POS = ["adj.", "nomen.", "verben."]

##################################################################
# Methods
def print_synsets(a_synsets, a_stream = sys.stdout):
    """
    Output synsets

    @param a_synsets - dictionary of synsets
    @param a_stream - output stream

    @return \c void
    """
    pass

def parse_relfile(a_fname, a_relations):
    """
    Parse GemaNet relations file

    @param a_fname - name of input file
    @param a_relations - dictionary for storing relations

    @return \c void
    """
    ikey = None
    idoc = ET.parse(a_fname).getroot()
    for rel_type in ("con_rel", "lex_rel"):
        for irel in idoc.iterfind(rel_type):
            ikey = (irel.get("from"), irel.get("to"))
            if ikey in a_relations:
                a_relations[ikey].add(irel.get("name"))
            else:
                a_relations[ikey] = set([irel.get("name")])

def parse_synset_file(a_fname, a_synid2examples, a_lexid2synid):
    """
    Parse GemaNet XML file

    @param a_fname - name of input file
    @param a_synid2examples - dictionary mapping synset id's to examples
    @param a_lexid2synid - dictionary mapping lexeme id's to synset id's

    @return dictionar
    """
    lex_id = ""; syn_id = ""
    iparaphrase = ilex = iform = None
    idoc = ET.parse(a_fname).getroot()
    for isynset in idoc.iterfind("synset"):
        syn_id = isynset.get("id")
        iparaphrase = isynset.find("./paraphrase")
        assert syn_id not in a_synid2examples, \
            "Duplicate description for synset {:s}".format(syn_id)
        a_synid2examples[syn_id] = (("" if iparaphrase is None else iparaphrase.text), \
                                        [el.text for el in isynset.iterfind(".//example")])
        for ilex in isynset.iterfind("./lexUnit"):
            lex_id = ilex.get("id")
            for iform in ilex.iterfind("./orthForm"):
                form = iform.text
                if form in a_lexid2synid:
                    a_lexid2synid[form][0].add(lex_id)
                    a_lexid2synid[form][-1].append(syn_id)
                else:
                    a_lexid2synid[form] = (set([lex_id]), [syn_id])

def main(a_argv):
    """
    Main method for converting GermaNet XML to single tsv file.

    @param a_argv - command line arguments

    @return \c 0 on success, non-\c 0 otherwise
    """
    argparser = argparse.ArgumentParser(description = "Script for converting GemaNet XML to TSV format.")
    argparser.add_argument("gnet_dir", help = "directory containing GermaNet files")
    args = argparser.parse_args(a_argv)

    if not os.path.isdir(args.gnet_dir) or not os.access(args.gnet_dir, os.R_OK):
        raise RuntimeError("Can't read from directory: {:s}".format(args.gnet_dir))

    # Dictionary holding synset files.  This dictionary has the following structure:
    # syn_id2examples[synset_id] => (paraphrase, *examples)
    # lex_id2syn_id[lex_id] => (lex_id, *synset_id's)
    synid2examples = dict(); lexid2synid = dict();
    for ifile in chain.from_iterable(glob.iglob(os.path.join(args.gnet_dir, ipos + '*')) for ipos in POS):
        parse_synset_file(ifile, synid2examples, lexid2synid)

    relations = dict();
    parse_relfile(os.path.join(args.gnet_dir, "gn_relations.xml"), relations)
    return 0

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
