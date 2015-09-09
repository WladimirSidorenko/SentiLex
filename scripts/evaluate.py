#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Script for evaluating sentiment lexicon on test corpus.

USAGE:
evaluate.py [lemma_file] sentiment_lexicon test_corpus_dir/

"""

##################################################################
# Libraries
from trie import SPACE_RE, Trie

from collections import defaultdict
import argparse
import codecs
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

##################################################################
# Constants and Variables

WSPAN_PREFIX = "word_"
WSPAN_PREFIX_RE = re.compile(WSPAN_PREFIX)
# regexp matching annotation spans that encompass single word
WSPAN = re.compile("{:s}(\d+)\Z".format(WSPAN_PREFIX), re.IGNORECASE)
# regexp matching annotation spans that encompass multiple words
WMULTISPAN  = re.compile("{:s}(\d+)..+{:s}(\d+)".format(WSPAN_PREFIX, \
                                                            WSPAN_PREFIX), \
                             re.IGNORECASE)

COMMA_SEP = ','
TAB_RE = re.compile("[ ]*\t+[ ]*")

ENCODING = "utf-8"             # default encoding of input files
WORD = "word"
POLARITY = "polarity"
MARKABLE = "markable"
MMAX_LEVEL = "mmax_level"
EMOEXPRESSION = "emo-expression"
WORDS_PTRN = "*.words.xml"     # globbing pattern for word files
WORDS_PTRN_RE = re.compile(".words.xml")
MRKBL_PTRN = "_emo-expression_level.xml"     # globbing pattern for annotation files

POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"
KNOWN_POLARITIES = set([POSITIVE, NEGATIVE, NEUTRAL])

TRUE_POS = 0                   # index of true positive counts
FALSE_POS = 1                  # index of false positive counts
FALSE_NEG = 2                  # index of false negative counts
COMMENT_RE = re.compile("(?:\A|\s+)# .*$")

##################################################################
# Methods
def _parse_span(ispan, a_int_fmt = False):
    """Generate and return a list of all word ids encompassed by ispan."""
    ret = []
    # split span on commas
    spans = ispan.split(COMMA_SEP)
    for s in spans:
        if WSPAN.match(s):
            if a_int_fmt:
                ret.append(int(WSPAN_PREFIX_RE.sub("", s)))
            else:
                ret.append(s)
        else:
            mobj = WMULTISPAN.match(s)
            if mobj:
                start, end = int(mobj.group(1)), int(mobj.group(2)) + 1
                if a_int_fmt:
                    ret += [w_id for w_id in xrange(start, end)]
                else:
                    ret += [(WSPAN_PREFIX + str(w_id)) for w_id in xrange(start, end)]
            else:
                raise ValueError("Unrecognized span format: {:s}".format(ispan))
    return ret

def _read_file(a_lexicon, a_fname, a_insert, a_enc = ENCODING):
    """
    General method for reading tab-separated files

    @param a_lexicon - lexicon to be populated
    @param a_fname - name of the file containign sentiment lexicon
    @param a_insert - custom insert function
    @param a_enc - encoding of the input file

    @return \c void
    """
    item1 = item2 = None
    with codecs.open(a_fname, 'r', a_enc) as ifile:
        for iline in ifile:
            iline = iline.strip()
            iline = COMMENT_RE.sub("", iline)
            if not iline:
                continue
            iline.lower()
            try:
                item1, item2 = TAB_RE.split(iline)
            except ValueError:
                print >> sys.stderr, "Invalid line format: '{:s}'".format(iline).encode(a_enc)
                raise
            a_insert(a_lexicon, item1, item2)

def _compute_fscore(a_stat):
    """
    Compute macro- and micro-averaged F-scores

    @param a_stat - statistics disctionary on single classes

    @return 2-tuple of macro- and micro-averaged F-scores
    """
    print repr(a_stat)
    sys.exit(66)

def _compute(a_lexicon, a_id_tok):
    """
    Compute macro- and micro-averaged F-scores for single file

    @param a_lexicon - lexicon whose quality should be testedи
    @param a_id_tok - sequence of annotated tokens extracted from file

    @return 6-tuple with the number of correct and wrong matches,
    total tokens, as well as macro- and micro-averaged F-scores
    """
    macro_F1 = micro_F1 = 0.
    correct_toks = wrong_toks = 0
    total_toks = sum([len(itok[-1]) or 1 for itok in a_id_tok])
    # dictionary used for counting correct and wrong matches of each
    # class
    stat = defaultdict(lambda: [0, 0, 0])
    # iterate over tokens and update statistics accordingly
    hasmatch = False
    for i, (_, iform, ilemma, ianno) in enumerate(a_id_tok[:40]):
        hasmatch = a_lexicon.match([iform, ilemma])
        # check cases when the lexicon actually matched
        if hasmatch:
            for astate in a_lexicon.active_states:
                istate, istart, iend = astate
                if not istate.final:
                    continue
                for mclass in istate.classes:
                    if (istart, mclass) in ianno:
                        stat[mclass][TRUE_POS] += 1
                        correct_toks += 1
                        ianno.pop((istart, mclass))
                    else:
                        stat[mclass][FALSE_POS] += 1
                        wrong_toks += 1
            wrong_toks += len(ianno)
            for _, iclass in ianno:
                stat[iclass][FALSE_NEG] += 1
        elif not ianno:
            correct_toks += 1
            stat[NEUTRAL][TRUE_POS] += 1
        else:
            wrong_toks += len(ianno)
            for _, iclass in ianno:
                stat[iclass][FALSE_NEG] += 1
            stat[NEUTRAL][FALSE_POS] += 1
        # let Trie proceed to the next state
        a_lexicon.match([' ', None])
    macro_F1, micro_F1 = _compute_fscore(stat)
    return (correct_toks, wrong_toks, total_toks, macro_F1, micro_F1)

def eval_lexicon(a_lexicon, a_base_dir, a_anno_dir, a_form2lemma):
    """
    Evaluate sentiment lexicon on a real corpus

    @param a_lexicon - lexicon to test (as a Trie)
    @param a_base_dir - directory containing base files of the MMAX project
    @param a_anno_dir - directory containing annotation files of the MMAX project
    @param a_form2lemma - dictionary mapping word forms to lemmas

    @return 6-tuple with macro- and micro-averaged precision, recall, and F-measure
    """
    itok = ""
    id_tok = []
    wid2tid = dict()
    stat = []; macro_F1 = []; micro_F1 = []
    wid = tid = imacro_F1 = imicro_F1 = icorrect = iwrong = itotal = -1
    annofname = idoc = ispan = wid = None
    # iterate over
    for basefname in glob.iglob(os.path.join(a_base_dir, WORDS_PTRN)):
        if not os.access(basefname, os.R_OK):
            continue
        annofname = os.path.join(a_anno_dir, \
                                     os.path.basename(WORDS_PTRN_RE.sub("", basefname) + \
                                                          MRKBL_PTRN))
        if not os.path.exists(annofname) or not os.access(annofname, os.R_OK):
            print >> sys.stderr, "Cannot read annotation file '{:s}'".format(annofname)
            continue
        # read tokens
        wid2tid.clear(); del id_tok[:]
        idoc = ET.parse(basefname).getroot()
        for iword in idoc.iter(WORD):
            wid = iword.attrib["id"]
            itok = SPACE_RE.sub(' ', iword.text.strip()).lower()
            wid2tid[wid] = len(id_tok)
            id_tok.append((wid, itok, a_form2lemma[itok] if itok in a_form2lemma else None, []))
        # enrich tokens with annotations
        idoc = ET.parse(annofname).getroot()
        for ianno in idoc:
            assert ianno.get(MMAX_LEVEL, "").lower() == EMOEXPRESSION, \
                "Invalid element specified as annotation"
            ipolarity = ianno.get(POLARITY)
            assert ipolarity in KNOWN_POLARITIES, "Unknown polarity value: '{:s}'".format(ipolarity)
            ispan = _parse_span(ianno.get("span"))
            tid = wid2tid[ispan[-1]]
            print "tid =", repr(tid)
            print "ipolarity =", repr(ipolarity)
            # add respective class only to the last term in annotation
            # sequence, but remember which tokens this annotation
            # covered
            id_tok[tid][-1].append(([wid2tid[iwid] for iwid in ispan], ipolarity))
        # now, do the actual computation of matched items
        icorrect, iwrong, itotal, imacro_F1, imicro_F1 = _compute(a_lexicon, id_tok)
        stat.append((icorrect, iwrong, itotal))
        macro_F1.append(imacro_F1)
        micro_F1.append(imicro_F1)

def main(argv):
    """
    Main method for estimating quality of a sentiment lexicon

    @param argv - CLI arguments

    @return 0 on success, non-0 otherwise
    """
    # parse arguments
    argparser = argparse.ArgumentParser(description = \
                                        """Script for evaluating sentiment lexicon on test corpus.""")
    argparser.add_argument("-e", "--encoding", help = "encoding of input files", type = str, \
                               default = ENCODING)
    argparser.add_argument("--lemma-file", help = "file containing lemmas of corpus words", \
                               type = str)
    argparser.add_argument("sentiment_lexicon", help = "sentiment lexicon to test", type = str)
    argparser.add_argument("corpus_base_dir", help = \
                           "directory containing word files of sentiment corpus in MMAX format", \
                               type = str)
    argparser.add_argument("corpus_anno_dir", help = \
                           "directory containing annotation files of sentiment corpus in MMAX format", \
                               type = str)
    args = argparser.parse_args(argv)
    # read-in lexicon
    ilex = Trie()
    _read_file(ilex, args.sentiment_lexicon, a_insert = lambda lex, wrd, cls: \
                   lex.add(wrd, cls) if cls != NEUTRAL and cls in KNOWN_POLARITIES else None)
    form2lemma = dict()
    if args.lemma_file is not None:
        _read_file(form2lemma, args.lemma_file, a_insert = \
                      lambda lex, form, lemma: lex.setdefault(form, lemma))
    # evaluate it on corpus
    eval_lexicon(ilex, args.corpus_base_dir, args.corpus_anno_dir, form2lemma)

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
