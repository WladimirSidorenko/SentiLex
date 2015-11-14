#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Script for adding missing polar terms to sentiment corpus.

USAGE:
adddic2corpus.py [lemma_file] sentiment_lexicon test_corpus_dir/

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from evaluate import insert_lex, is_word, parse_span, read_file, \
    ENCODING, EMOEXPRESSION, KNOWN_POLARITIES, MARKABLE, MMAX_LEVEL, \
    MRKBL_PTRN, POLARITY, WORD, WORDS_PTRN, WORDS_PTRN_RE
from trie import SPACE_RE, CONTINUE, Trie

from copy import deepcopy
import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

##################################################################
# Constants and Variables
ID_PRFX = "markable_500100"
ID_CNT = -1                     # counter of new markables
DIFF_EEXPRESSION = "diff-emo-expression"
DIFF_MRKBL_PTRN = "_{:s}_level.xml".format(DIFF_EEXPRESSION)
# XML namespace
MMAX_NS_URI = "www.eml.org/NameSpaces/emo-expression"
MMAX_NS = {"mmax": MMAX_NS_URI}
ID = "id"
SPAN = "span"
POLARITY = "polarity"
DIFF_TYPE = "diff_type"

XML_HEADER = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE markables SYSTEM "markables.dtd">
"""

##################################################################
# Methods
def _get_new_id():
    """Generate new unique markable id

    @return string representing new unique markable id

    """
    global ID_CNT; ID_CNT += 1
    return ID_PRFX + str(ID_CNT)

def _add_mrkbl(a_tree, a_attrs):
    """Add difference markable to XML tree

    @param a_tree - target XML tree to which new markable should be added
    @param a_attrs - attributes of the new markable

    @return \c void

    """
    # check if necessary attributes are present
    assert DIFF_TYPE in a_attrs, "Missing diff type for new diff markable."
    assert ID in a_attrs, "Missing id for new diff markable."
    assert POLARITY in a_attrs, "Missing polarity value for new diff markable."
    assert SPAN in a_attrs, "Missing span for new diff markable."
    # set default attribute values
    a_attrs.setdefault("intensity", "medium")
    a_attrs.setdefault("sarcasm", "false")
    a_attrs.setdefault("mmax_level", DIFF_EEXPRESSION)
    a_attrs.setdefault("sentiment_ref", "empty")
    # add new markable
    mrkbl = ET.SubElement(a_tree, MARKABLE, a_attrs)

def _add_lex(a_lexicon, a_id_tok, a_tree):
    """Add missing terms to XML tree with markables

    @param a_lexicon - trie containing polar terms
    @param a_id_tok - list of tuples containing word ids, tokens,
                      lemmas, and annotations
    @param a_tree - target tree to which new terms should be added

    @return \c void

    """
    matched_states = set()
    istart = iend = -1
    frzset = istate = None
    isneutral = hasmatch = False
    for i, (w_id, iform, ilemma, ianno) in enumerate(a_id_tok):
        print("iform: {:s}".format(iform).encode(ENCODING), file = sys.stderr)
        print("ilemma: {:s}".format(ilemma).encode(ENCODING), file = sys.stderr)
        isneutral = not bool(ianno)
        hasmatch = a_lexicon.match([iform, ilemma], a_start = i, a_reset = CONTINUE)
        print("hasmatch: {:s}".format(str(hasmatch)).encode(ENCODING), file = sys.stderr)
        if hasmatch:
            for astate in a_lexicon.active_states:
                istate, istart, iend = astate
                if not istate.final:
                    continue
                frzset = frozenset(istate.classes)
                # skip duplicate states that arise from using lemmas
                if (istart, frzset) in matched_states:
                    continue
                else:
                    matched_states.add((istart, frzset))
                for mclass in istate.classes:
                    if (istart, mclass) in ianno:
                        ianno.remove((istart, mclass))
                    else:
                        _add_mrkbl(a_tree, {DIFF_TYPE: "missing", ID: _get_new_id(), \
                                                POLARITY: mclass,  \
                                                SPAN: w_id if istart == i else \
                                                a_id_tok[istart][0] + ".." + w_id})
                        # ientry = ' '.join([t[1] for t in a_id_tok[istart:i+1]])
                        # print(">>> excessive: {:s} ({:s})".format(ientry, mclass).encode(ENCODING), \
                        #           file = sys.stderr)
            if ianno:
                # print("did not match iform = {:s} ({:s})".format(iform, repr(ianno)), \
                #           file = sys.stderr)
                for istart, iclass in ianno:
                    _add_mrkbl(a_tree, {DIFF_TYPE: "redundant", ID: _get_new_id(), \
                                            POLARITY: iclass,  \
                                            SPAN: w_id if istart == i else \
                                            a_id_tok[istart][0] + ".." + w_id})
                    # ientry = ' '.join([t[1] for t in a_id_tok[istart:i+1]])
                    # print("<<< missing: {:s} ({:s})".format(ientry, iclass).encode(ENCODING), \
                    #           file = sys.stderr)
            matched_states.clear()
        elif not isneutral:
            for istart, iclass in ianno:
                _add_mrkbl(a_tree, {DIFF_TYPE: "redundant", ID: _get_new_id(), \
                                        POLARITY: iclass,  \
                                        SPAN: w_id if istart == i else \
                                        a_id_tok[istart][0] + ".." + w_id})
                # ientry = ' '.join([t[1] for t in a_id_tok[istart:i+1]])
                # print("<<< missing: {:s} ({:s})".format(ientry, iclass).encode(ENCODING), \
                #           file = sys.stderr)
        # let Trie proceed to the next state
        a_lexicon.match([' ', None], a_reset = CONTINUE)
        # print("stat =", repr(stat), file = sys.stderr)
    a_lexicon.match((None, None)) # reset active states

def _dcopy_emo_xml(a_srctree):
    """Create a pruned deep copy of annotation XML with emo-expressions

    @param a_srctree - source XML tree
    @param a_encoding - encoding of the output file

    @return pruned deep copy of source XML tree

    """
    ret = deepcopy(a_srctree)
    root = ret.getroot()
    # prune the tree
    for imrkbl in root.findall("mmax:markable", MMAX_NS):
        root.remove(imrkbl)
    return ret

def add_lexicon(a_lexicon, a_base_dir, a_anno_dir, a_form2lemma):
    """Add sentiment lexicon as markables to corpus

    @param a_lexicon - lexicon to test (as a Trie)
    @param a_base_dir - directory containing base files of the MMAX project
    @param a_anno_dir - directory containing annotation files of the MMAX project
    @param a_form2lemma - dictionary mapping word forms to lemmas

    @return \c void

    """
    id_tok = []
    wid = tid = -1
    wid2tid = dict()
    idoc = iroot = None
    annofname = diff_fname = ""

    ET.register_namespace('', MMAX_NS_URI)
    for basefname in glob.iglob(os.path.join(a_base_dir, WORDS_PTRN)):
        print("Processing file '{:s}'".format(basefname), file = sys.stderr)
        if not os.access(basefname, os.R_OK):
            continue
        annofname = os.path.join(a_anno_dir, \
                                     os.path.basename(WORDS_PTRN_RE.sub("", basefname) + \
                                                          MRKBL_PTRN))
        if not os.path.exists(annofname) or not os.access(annofname, os.R_OK):
            print("Cannot read annotation file '{:s}'".format(annofname), file = sys.stderr)
            continue
        # read tokens
        wid2tid.clear(); del id_tok[:]
        idoc = ET.parse(basefname).getroot()
        for iword in idoc.iter(WORD):
            wid = iword.attrib["id"]
            itok = SPACE_RE.sub(' ', iword.text.strip()).lower()
            wid2tid[wid] = len(id_tok)
            id_tok.append((wid, itok, a_form2lemma[itok] if itok in a_form2lemma else None, set()))
        # enrich tokens with annotations
        idoc = ET.parse(annofname)
        iroot = idoc.getroot()
        for ianno in iroot:
            assert ianno.get(MMAX_LEVEL, "").lower() == EMOEXPRESSION, \
                "Invalid element specified as annotation"
            ipolarity = ianno.get(POLARITY)
            assert ipolarity in KNOWN_POLARITIES, "Unknown polarity value: '{:s}'".format(ipolarity)
            ispan = parse_span(ianno.get("span"))
            tid = wid2tid[ispan[-1]]
            # add respective class only to the last term in annotation
            # sequence, but remember which tokens this annotation
            # covered
            if is_word(id_tok[tid][1]) or is_word(id_tok[tid][2]):
                # for discontinuous spans, simply adding start of span
                # (`wid2tid[ispan[0]`) might cause problems, but trie
                # does not match discontinuous spans anyway
                id_tok[tid][-1].add((wid2tid[ispan[0]], ipolarity))
        # create XML tree with difference markables
        diff_tree = _dcopy_emo_xml(idoc)
        # add new terms as difference markables to corpus
        _add_lex(a_lexicon, id_tok, diff_tree.getroot())
        # output generated XML tree to file
        diff_fname = os.path.join(a_anno_dir, \
                                     os.path.basename(WORDS_PTRN_RE.sub("", basefname) + \
                                                          DIFF_MRKBL_PTRN))
        with open(diff_fname, 'w') as ofile:
            ofile.write(XML_HEADER)
            diff_tree.write(ofile, encoding = "UTF-8", xml_declaration = False)
        sys.exit(66)

def main(argv):
    """
    Main method for adding missing polar terms to sentiment corpus.

    @param argv - CLI arguments

    @return 0 on success, non-0 otherwise

    """
    # parse arguments
    argparser = argparse.ArgumentParser(description = \
                                        "Script for adding missing polar terms to sentiment corpus.")
    argparser.add_argument("-l", "--lemma-file", help = "file containing lemmas of corpus words", \
                               type = str)
    argparser.add_argument("sentiment_lexicon", help = "sentiment lexicon to add", type = str)
    argparser.add_argument("corpus_base_dir", help = \
                           "directory containing word files of sentiment corpus in MMAX format", \
                               type = str)
    argparser.add_argument("corpus_anno_dir", help = \
                           "directory containing annotation files of sentiment corpus in MMAX format", \
                               type = str)
    args = argparser.parse_args(argv)
    # read-in lexicon
    ilex = Trie(a_ignorecase = True)
    read_file(ilex, args.sentiment_lexicon, a_insert = insert_lex)
    form2lemma = dict()
    if args.lemma_file is not None:
        read_file(form2lemma, args.lemma_file, a_insert = \
                      lambda lex, form, lemma: lex.setdefault(form, lemma))
    # evaluate it on corpus
    add_lexicon(ilex, args.corpus_base_dir, args.corpus_anno_dir, form2lemma)

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
