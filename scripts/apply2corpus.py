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

from evaluate import read_file, parse_span, \
    ENCODING, WORDS_PTRN, WORDS_PTRN_RE, MRKBL_PTRN
from trie import SPACE_RE, CONTINUE, Trie

##################################################################
# Constants and Variables
DIFF_MRKBL_PTRN = "_diff-emo-expression_level.xml"

##################################################################
# Methods
def _add_lex(argv):
    """

    """
    pass

def _dcopy_emo_xml(a_srctree):
    """Create a pruned deep copy of annotation XML with emo-expressions

    @param a_srctree - source XML tree
    @param a_encoding - encoding of the output file

    @return pruned deep copy of source XML tree

    """
    ret = deepcopy(a_srctree)
    # prune the tree
    for imrkbl in ret.iterfind():
        ret.remove(imrkbl)
    return ret

def add_lexicon(a_lexicon, a_id_tok, a_pr_stat, a_fscore_stat):
    """Add sentiment lexicon as markables to corpus

    @param a_lexicon - lexicon to test (as a Trie)
    @param a_base_dir - directory containing base files of the MMAX project
    @param a_anno_dir - directory containing annotation files of the MMAX project
    @param a_form2lemma - dictionary mapping word forms to lemmas
    @param a_output_errors - boolean flag indicating whether dictionary errors
                        should be printed
    @param a_encoding - encoding of the output file

    @return \c void

    """
    id_tok = []
    wid = tid = -1
    annofname = diff_fname = ""
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
        idoc = ET.parse(annofname).getroot()
        for ianno in idoc:
            assert ianno.get(MMAX_LEVEL, "").lower() == EMOEXPRESSION, \
                "Invalid element specified as annotation"
            ipolarity = ianno.get(POLARITY)
            assert ipolarity in KNOWN_POLARITIES, "Unknown polarity value: '{:s}'".format(ipolarity)
            ispan = _parse_span(ianno.get("span"))
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
        diff_tree = _dcopy_emo_xml(idoc, a_encoding)
        # output generated XML tree to file
        diff_fname = os.path.join(a_anno_dir, \
                                     os.path.basename(WORDS_PTRN_RE.sub("", basefname) + \
                                                          DIFF_MRKBL_PTRN))
        with open(diff_fname, 'w') as ofile:
            diff_tree.write(ofile)

def main(argv):
    """
    Main method for adding missing polar terms to sentiment corpus.

    @param argv - CLI arguments

    @return 0 on success, non-0 otherwise

    """
    # parse arguments
    argparser = argparse.ArgumentParser(description = \
                                        "Script for adding missing polar terms to sentiment corpus.")
    argparser.add_argument("-e", "--encoding", help = "encoding of lexicon files", type = str, \
                               default = ENCODING)
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
    read_file(ilex, args.sentiment_lexicon, a_insert = _insert_lex)
    form2lemma = dict()
    if args.lemma_file is not None:
        read_file(form2lemma, args.lemma_file, a_insert = \
                      lambda lex, form, lemma: lex.setdefault(form, lemma))
    # evaluate it on corpus
    add_lexicon(ilex, args.corpus_base_dir, args.corpus_anno_dir, form2lemma, args.encoding)

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
