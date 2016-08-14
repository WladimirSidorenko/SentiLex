#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""Script for evaluating sentiment lexicon on test corpus.

USAGE:
evaluate.py [lemma_file] sentiment_lexicon corpus_basedata_dir/ \
            corpus_basedata_dir/

"""

##################################################################
# Libraries
from __future__ import print_function, unicode_literals
from trie import SPACE_RE, CONTINUE, Trie

from collections import defaultdict
import argparse
import codecs
import glob
import numpy as np
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
WMULTISPAN = re.compile("{:s}(\d+)..+{:s}(\d+)".format(
    WSPAN_PREFIX, WSPAN_PREFIX), re.IGNORECASE)

COMMA_SEP = ','
TAB_RE = re.compile("[ ]*\t+[ ]*")
WORD_RE = re.compile("#?[-\w]*\w[-\w]*$", re.U)

ENCODING = "utf-8"             # default encoding of input files
WORD = "word"
POLARITY = "polarity"
MARKABLE = "markable"
MMAX_LEVEL = "mmax_level"
EMOEXPRESSION = "emo-expression"
WORDS_PTRN = "*.words.xml"     # globbing pattern for word files
WORDS_PTRN_RE = re.compile(".words.xml")
# globbing pattern for annotation files
MRKBL_PTRN = "_emo-expression_level.xml"

POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"
KNOWN_POLARITIES = set([POSITIVE, NEGATIVE, NEUTRAL])

TRUE_POS = 0                   # index of true positive counts
FALSE_POS = 1                  # index of false positive counts
FALSE_NEG = 2                  # index of false negative counts
COMMENT_RE = re.compile("(?:\A|\s+)#(?:\s|#).*$")

PRECISION = 0                   # index of precision field
RECALL = 1                      # index of recall field


##################################################################
# Methods
def insert_lex(a_lex, a_wrd, a_cls):
    """
    Insert new word into polarity lexicon after checking its polarity class


    @param a_lex - lexicon in which new term should be inserted
    @param a_wrd - polar term to insert
    @param a_cls - polarity class to check

    @return \c void (an Exception is raised if the class is unknown)
    """
    if a_cls in KNOWN_POLARITIES:
        if a_cls != NEUTRAL:
            a_lex.add(a_wrd, a_cls)
    else:
        raise RuntimeError("Unrecognized polarity class: {:s}".format(
            repr(a_cls)))


def is_word(a_word):
    """
    Check if given string forms a valid word

    @param a_word - string to be checked

    @return \c true if this string consists of alphanumeric characters
    """
    return (a_word is not None and (WORD_RE.match(a_word) is not None))


def parse_span(ispan, a_int_fmt=False):
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
                    ret += [(WSPAN_PREFIX + str(w_id))
                            for w_id in xrange(start, end)]
            else:
                raise ValueError(
                    "Unrecognized span format: {:s}".format(ispan))
    return ret


def read_file(a_lexicon, a_fname, a_insert, a_enc=ENCODING):
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
            iline = COMMENT_RE.sub("", iline)
            iline = iline.strip()
            if not iline:
                continue
            iline.lower()
            try:
                item1, item2 = TAB_RE.split(iline)
            except ValueError:
                print(
                    "Invalid line format: '{:s}'".format(iline).encode(a_enc),
                    file=sys.stderr)
                raise
            a_insert(a_lexicon, item1, item2)


def _compute_fscores(a_stat, a_fscore_stat):
    """
    Compute macro- and micro-averaged F-scores

    @param a_stat - statistics disctionary on single classes
    @param a_fscore_stat - verbose statistics with F-scores for each
                      particular class (will be updated in this method)

    @return 6-tuple with macro- and micro-averaged precision, recall,
      and F-scores
    """
    macro_P = micro_P = macro_R = micro_R = macro_F1 = micro_F1 = 0.
    n_classes = len(a_stat)
    if not n_classes:
        return (0., 0.)
    macro_F1 = micro_F1 = iF1 = iprec = 0.
    total_tp = total_fp = total_fn = 0
    # obtain statistics for all classes
    for iclass, (tp, fp, fn) in a_stat.iteritems():
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if tp or (fp and fn):
            iprec = tp / float(tp + fp) if (tp or fp) else 0.
            ircall = tp / float(tp + fn) if (tp or fn) else 0.
            if iprec or ircall:
                iF1 = 2 * (iprec * ircall) / (iprec + ircall)
                macro_P += iprec
                macro_R += ircall
                macro_F1 += iF1
            else:
                iF1 = 0
        else:
            iF1 = 0
        a_fscore_stat[iclass].append(iF1)
    # compute macro- and micro-averaged scores
    macro_P /= float(n_classes)
    macro_R /= float(n_classes)
    macro_F1 /= float(n_classes)
    if total_tp or (total_fp and total_fn):
        micro_P = total_tp / float(total_tp + total_fp)
        micro_R = total_tp / float(total_tp + total_fn)
        micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R)
    return (macro_P, micro_P, macro_R, micro_R, macro_F1, micro_F1)


def _compute(a_lexicon, a_id_tok, a_pr_stat, a_fscore_stat,
             a_output_errors, a_full_corpus=False):
    """Compute macro- and micro-averaged F-scores for single file

    @param a_lexicon - lexicon whose quality should be tested
    @param a_id_tok - sequence of annotated tokens extracted from file
    @param a_pr_stat - verbose statistics with precision and recall
    @param a_fscore_stat - verbose statistics with F-scores for each
                      particular class (will be updated in this method)
    @param a_output_errors - boolean flag indicating whether dictionary errors
                        should be printed
    @param a_full_corpus - compute scores on the full corpus

    @return 6-tuple with macro- and micro-averaged precision, recall,
    and F-scores

    """
    # dictionary used for counting correct and wrong matches of each
    # class
    stat = defaultdict(lambda: [0, 0, 0])
    # iterate over tokens and update statistics accordingly
    ientry = frzset = None
    matched_states = set()
    hasmatch = isneutral = False
    for i, (_, iform, ilemma, ianno) in enumerate(a_id_tok):
        # print("iform =", repr(iform), file = sys.stderr)
        # print("ianno =", repr(ianno), file = sys.stderr)
        isneutral = not bool(ianno)
        hasmatch = a_lexicon.match([iform, ilemma],
                                   a_start=i, a_reset=CONTINUE)
        # print("hasmatch =", repr(hasmatch), file = sys.stderr)
        # check cases when the lexicon actually matched
        if hasmatch:
            # print("a_lexicon.active_states =", repr(a_lexicon.active_states),
            #           file = sys.stderr)
            for astate in a_lexicon.active_states:
                istate, istart, _ = astate
                # print("istate, istart =", repr(istate), repr(istart),
                #           file = sys.stderr)
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
                        # print("matched iform = {:s} ({:s}) with {:s}".format(
                        #         repr(iform), repr(ianno), repr(mclass)),
                        #           file = sys.stderr)
                        stat[mclass][TRUE_POS] += 1
                        ianno.remove((istart, mclass))
                    else:
                        stat[mclass][FALSE_POS] += 1
                        if a_output_errors:
                            ientry = ' '.join([t[1]
                                               for t in a_id_tok[istart:i+1]])
                            print(">>> excessive: {:s} ({:s})".format(
                                ientry, mclass).encode(ENCODING),
                                file=sys.stderr)
                        if istart != i:
                            stat[NEUTRAL][FALSE_NEG] += 1
            if ianno:
                # print("did not match iform = {:s} ({:s})".format(iform,
                # repr(ianno)), file=sys.stderr)
                for istart, iclass in ianno:
                    stat[iclass][FALSE_NEG] += 1
                    if a_output_errors:
                        ientry = ' '.join([t[1] for t in a_id_tok[istart:i+1]])
                        print("<<< missing: {:s} ({:s})".format(
                            ientry, iclass).encode(ENCODING), file=sys.stderr)
            elif isneutral:
                stat[NEUTRAL][FALSE_NEG] += 1
                if a_output_errors:
                    print("<<< missing: {:s} ({:s})".format(
                        a_id_tok[i][1], NEUTRAL).encode(ENCODING),
                        file=sys.stderr)
            matched_states.clear()
        elif isneutral:
            stat[NEUTRAL][TRUE_POS] += 1
        else:
            stat[NEUTRAL][FALSE_POS] += 1
            for istart, iclass in ianno:
                stat[iclass][FALSE_NEG] += 1
                if a_output_errors:
                    ientry = ' '.join([t[1] for t in a_id_tok[istart:i+1]])
                    print("<<< missing: {:s} ({:s})".format(
                        ientry, iclass).encode(ENCODING), file=sys.stderr)
        # let Trie proceed to the next state
        a_lexicon.match([' ', None], a_reset=CONTINUE)
        # print("stat =", repr(stat), file = sys.stderr)
    a_lexicon.match((None, None))  # reset active states
    # update statistics
    if a_full_corpus:
        return stat
    for c, cstat in stat.iteritems():
        if cstat[TRUE_POS]:
            a_pr_stat[c][PRECISION].append(cstat[TRUE_POS] /
                                           float(cstat[TRUE_POS] +
                                                 cstat[FALSE_POS]))
            a_pr_stat[c][RECALL].append(cstat[TRUE_POS] /
                                        float(cstat[TRUE_POS] +
                                              cstat[FALSE_NEG]))
        else:
            a_pr_stat[c][PRECISION].append(0.)
            a_pr_stat[c][RECALL].append(0.)
    # print("stat =", repr(stat), file = sys.stderr)
    return _compute_fscores(stat, a_fscore_stat)


def eval_lexicon(a_lexicon, a_base_dir, a_anno_dir,
                 a_form2lemma, a_output_errors, a_full_corpus=False):
    """Evaluate sentiment lexicon on a real corpus.

    @param a_lexicon - lexicon to test (as a Trie)
    @param a_base_dir - directory containing base files of the MMAX project
    @param a_anno_dir - directory containing annotation files of the MMAX
      project
    @param a_form2lemma - dictionary mapping word forms to lemmas
    @param a_output_errors - boolean flag indicating whether dictionary errors
                        should be printed
    @param a_full_corpus - compute scores on the full corpus

    @return 6-tuple with macro- and micro-averaged precision,
      recall, and F-measure

    """
    itok = ""
    id_tok = []
    wid2tid = dict()
    pr_stat = defaultdict(lambda: [[], []])
    fscore_stat = defaultdict(list)
    macro_F1 = []
    micro_F1 = []
    macro_P = []
    micro_P = []
    macro_R = []
    micro_R = []
    wid = tid = imacro_F1 = imicro_F1 = -1
    annofname = full_stat = trg_stat = idoc = ispan = wid = None
    if a_full_corpus:
        full_stat = defaultdict(lambda: [0, 0, 0])
    # iterate over
    for basefname in glob.iglob(os.path.join(a_base_dir, WORDS_PTRN)):
        print("Processing file '{:s}'".format(basefname), file=sys.stderr)
        if not os.access(basefname, os.R_OK):
            continue
        annofname = os.path.join(a_anno_dir,
                                 os.path.basename(
                                     WORDS_PTRN_RE.sub("", basefname) +
                                     MRKBL_PTRN))
        if not os.path.exists(annofname) or not os.access(annofname, os.R_OK):
            print("Cannot read annotation file '{:s}'".format(annofname),
                  file=sys.stderr)
            continue
        # read tokens
        wid2tid.clear()
        del id_tok[:]
        idoc = ET.parse(basefname).getroot()
        for iword in idoc.iter(WORD):
            wid = iword.attrib["id"]
            itok = SPACE_RE.sub(' ', iword.text.strip()).lower()
            wid2tid[wid] = len(id_tok)
            id_tok.append((wid, itok, a_form2lemma[itok]
                           if itok in a_form2lemma else None, set()))
        # enrich tokens with annotations
        idoc = ET.parse(annofname).getroot()
        for ianno in idoc:
            assert ianno.get(MMAX_LEVEL, "").lower() == EMOEXPRESSION, \
                "Invalid element specified as annotation"
            ipolarity = ianno.get(POLARITY)
            assert ipolarity in KNOWN_POLARITIES, \
                "Unknown polarity value: '{:s}'".format(ipolarity)
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

        # now, do the actual computation of matched items
        if a_full_corpus:
            cstat = _compute(a_lexicon, id_tok, pr_stat, fscore_stat,
                             a_output_errors, a_full_corpus)
            for k, v in cstat.iteritems():
                trg_stat = full_stat[k]
                for i, j in enumerate(v):
                    trg_stat[i] += j
        else:
            imacro_P, imicro_P, imacro_R, imicro_R, imacro_F1, imicro_F1 = \
                _compute(a_lexicon, id_tok, pr_stat, fscore_stat,
                         a_output_errors, a_full_corpus)
            macro_P.append(imacro_P)
            micro_P.append(imicro_P)
            macro_R.append(imacro_R)
            micro_R.append(imicro_R)
            macro_F1.append(imacro_F1)
            micro_F1.append(imicro_F1)

    print("{:15s}{:>20s}{:>20s}{:>24s}".format(
        "Class", "Precision", "Recall", "F-score"), file=sys.stderr)
    iprec = ircall = fscore = 0.
    if a_full_corpus:
        total_tp = total_fp = total_fn = 0
        for iclass, (tp, fp, fn) in full_stat.iteritems():
            total_tp += tp
            total_fp += fp
            total_fn += fn
            iprec = tp / float(tp + fp) if (tp or fp) else 0.
            macro_P.append(iprec)
            ircall = tp / float(tp + fn) if (tp or fn) else 0.
            macro_R.append(ircall)
            fscore = 2 * iprec * ircall / ((iprec + ircall) or 1.)
            macro_F1.append(fscore)
            print("{:15s}{:>20%} {:>20%}"
                  " {:>24%}".format(iclass, iprec, ircall,
                                    fscore))
        print("{:15s}{:>20%} {:>20%}"
              " {:>24%}".format(
                  "Macro-average", np.mean(macro_P), np.mean(macro_R),
                  np.mean(macro_F1)))
        micro_P = total_tp / float(total_tp + total_fp) \
            if (total_tp or total_fp) else 0.
        micro_R = total_tp / float(total_tp + total_fn) \
            if (total_tp or total_fn) else 0.
        micro_F1 = 2 * micro_P * micro_R / ((micro_P + micro_R) or 1.)
        print("{:15s}{:>20%} {:>20%}"
              " {:>24%}".format(
                  "Micro-average", micro_P, micro_R, micro_F1))
    else:
        for iclass, fscores in fscore_stat.iteritems():
            iprec, ircall = pr_stat[iclass][PRECISION], pr_stat[iclass][RECALL]
            print("{:15s}{:>10.2%} (+/- {:6.2%}){:>10.2%}"
                  " (+/- {:6.2%}){:>10.2%} (+/- {:6.2%})".format(
                      iclass, np.mean(iprec), np.std(iprec), np.mean(ircall),
                      np.std(ircall), np.mean(fscores), np.std(fscores)))
        print("{:15s}{:>10.2%} (+/- {:6.2%}){:>10.2%}"
              " (+/- {:6.2%}){:>10.2%} (+/- {:6.2%})".format(
                  "Macro-average", np.mean(macro_P), np.std(macro_P),
                  np.mean(macro_R), np.std(macro_R),
                  np.mean(macro_F1), np.std(macro_F1)))
        print("{:15s}{:>10.2%} (+/- {:6.2%}){:>10.2%}"
              " (+/- {:6.2%}){:>10.2%} (+/- {:6.2%})".format(
                  "Micro-average", np.mean(micro_P), np.std(micro_P),
                  np.mean(micro_R), np.std(micro_R),
                  np.mean(micro_F1), np.std(micro_F1)))


def main(argv):
    """Main method for estimating quality of a sentiment lexicon.

    @param argv - CLI arguments

    @return 0 on success, non-0 otherwise

    """
    # parse arguments
    argparser = argparse.ArgumentParser(description="Script for evaluating"
                                        " sentiment lexicon on test corpus.")
    argparser.add_argument("-e", "--encoding", help="encoding of input files",
                           type=str, default=ENCODING)
    argparser.add_argument("-f", "--full",
                           help="compute scores on the full corpus",
                           action="store_true")
    argparser.add_argument("-l", "--lemma-file",
                           help="file containing lemmas of corpus words",
                           type=str)
    argparser.add_argument("-v", "--verbose",
                           help="output missing and excessive terms",
                           action="store_true")
    argparser.add_argument("sentiment_lexicon",
                           help="sentiment lexicon to test", type=str)
    argparser.add_argument("corpus_base_dir",
                           help="directory containing word files of sentiment"
                           " corpus in MMAX format", type=str)
    argparser.add_argument("corpus_anno_dir", help="directory containing"
                           " annotation files of sentiment corpus in MMAX"
                           " format",
                           type=str)
    args = argparser.parse_args(argv)
    # read-in lexicon
    ilex = Trie(a_ignorecase=True)
    read_file(ilex, args.sentiment_lexicon, a_insert=insert_lex)
    form2lemma = dict()
    if args.lemma_file is not None:
        read_file(form2lemma, args.lemma_file,
                  a_insert=lambda lex, form, lemma:
                  lex.setdefault(form, lemma))
    # evaluate it on corpus
    eval_lexicon(ilex, args.corpus_base_dir, args.corpus_anno_dir, form2lemma,
                 args.verbose, args.full)

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
