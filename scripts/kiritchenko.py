#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Kiritchenko et al.'s method (2014).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import ENCODING, ESC_CHAR, FMAX, FMIN, \
    INFORMATIVE_TAGS, NEGATIVE, POSITIVE, SENT_END_RE, \
    TAB_RE, NONMATCH_RE, MIN_TOK_CNT, check_word, normalize

from collections import defaultdict
from math import log
import codecs
import numpy as np
import sys


##################################################################
# Constants
FASTMODE = False
POS_IDX = 0
NEG_IDX = 1
NEUT_IDX = 2


##################################################################
# Methods
def _update_stat(a_tok_stat, a_tweet_stat, a_lemmas,
                 a_pos, a_neg, a_neut,
                 a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE):
    """Update statistics on occurrences of words containing seed terms.

    @param a_tok_stat - statistics on term occurrences
    @param a_tweet_stat - statistics on tweets
    @param a_lemmas - lemmas found in tweet
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return void

    @note modifies `a_tok_stat' and `a_tweet_stat' in place

    """
    tweet = ' '.join(sorted(a_lemmas))
    idx = -1
    if not a_lemmas:
        return
    elif a_lemmas & a_pos or a_pos_re.search(tweet):
        idx = POS_IDX
    elif a_lemmas & a_neg or a_neg_re.search(tweet):
        idx = NEG_IDX
    elif a_lemmas & a_neut:
        idx = NEUT_IDX
    if idx > -1:
        a_tweet_stat[idx] += 1
        for ilemma in a_lemmas:
            a_tok_stat[ilemma][idx] += 1
    a_lemmas.clear()


def _prune_stat(a_tok_stat):
    """Remove words with fewer occurrences than the minimum threshold.

    @param a_tok_stat - statistics on term occurrences

    @return \c void

    @note modifies `a_tok_stat' in place

    """
    w2delete = set()
    for w, cnts in a_tok_stat.iteritems():
        if sum(cnts) < MIN_TOK_CNT:
            w2delete.add(w)
    for w in w2delete:
        del a_tok_stat[w]


def _read_files(a_stat, a_crp_files, a_pos, a_neg, a_neut,
                a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE):
    """Read corpus files and populate one-directional co-occurrences.

    @param a_stat - statistics on term occurrences
    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return 2-tuple - number of positive and number of negative tweets

    @note modifies `a_stat' in place

    """
    print("Reading corpus...", end="", file=sys.stderr)
    i = 0
    iform = itag = ilemma = ""
    tlemmas = set()
    tweet_stat = [0, 0, 0]
    seeds = a_pos | a_neg | a_neut
    for ifname in a_crp_files:
        with codecs.open(ifname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip().lower()
                if iline and iline[0] == ESC_CHAR:
                    if FASTMODE:
                        i += 1
                        if i > 300:
                            break
                    _update_stat(a_stat, tweet_stat, tlemmas,
                                 a_pos, a_neg, a_neut,
                                 a_pos_re, a_neg_re)
                    continue
                elif not iline or SENT_END_RE.match(iline):
                    continue
                try:
                    iform, itag, ilemma = TAB_RE.split(iline)
                except:
                    print("Invalid line format at line: {:s}".format(
                        repr(iline)), file=sys.stderr
                    )
                    continue
                ilemma = normalize(ilemma)
                if a_pos_re.search(iform) or a_neg_re.search(iform) \
                   or iform in seeds:
                    tlemmas.add(iform)
                elif a_pos_re.search(ilemma) or a_neg_re.search(ilemma) \
                     or ilemma in seeds:
                    tlemmas.add(ilemma)
                elif itag[:2] not in INFORMATIVE_TAGS \
                        or not check_word(ilemma):
                    continue
                else:
                    tlemmas.add(ilemma)
            _update_stat(a_stat, tweet_stat, tlemmas,
                         a_pos, a_neg, a_neut,
                         a_pos_re, a_neg_re)
    print(" done", file=sys.stderr)
    # remove words with fewer occurrences than the minimum threshold
    _prune_stat(a_stat)
    return tweet_stat


def _stat2scores(a_stat, a_n_pos, a_n_neg, a_n_neut,
                 a_pos, a_neg, a_neut):
    """Convert statistics to scores.

    @param a_stat - statistics on terms
    @param a_n_pos - total number of positive tweets
    @param a_n_neg - total number of negative tweets
    @param a_n_neut - total number of neutral tweets
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded

    @return list of terms, their polarities, and scores

    """
    subj_score = pol_score = 0.
    ret = [(w, POSITIVE, FMAX) for w in a_pos] \
        + [(w, NEGATIVE, FMIN) for w in a_neg]
    n_subj = a_n_pos + a_n_neg
    for iterm, (ipos, ineg, ineut) in a_stat.iteritems():
        if iterm in a_pos or iterm in a_neg or iterm in a_neut:
            continue
        # decide whether the term is subjective or not
        subj_score = log(float(ipos + ineg) * a_n_neut /
                         (float(ineut * n_subj) or 1.) or 1., 2)
        # determine term's polarity
        if subj_score > 0:
            pol_score = log(float(ipos * a_n_neg) /
                            (float(ineg * a_n_pos) or 1.) or 1., 2)
            ret.append((iterm,
                        POSITIVE if pol_score > 0. else NEGATIVE,
                        pol_score + np.abs(subj_score)))
    return ret


def kiritchenko(a_N, a_crp_files, a_pos, a_neg, a_neut,
                a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE):
    """Method for generating sentiment lexicons using Kiritchenko's approach.

    @param a_N - number of terms to extract
    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return list of terms sorted according to their polarity scores

    """
    a_pos = set(normalize(w) for w in a_pos)
    a_neg = set(normalize(w) for w in a_neg)
    a_neut = set(normalize(w) for w in a_neut)

    stat = defaultdict(lambda: [0, 0, 0])
    n_pos, n_neg, n_neut = _read_files(stat, a_crp_files,
                                       a_pos, a_neg, a_neut,
                                       a_pos_re, a_neg_re)
    ret = _stat2scores(stat, n_pos, n_neg, n_neut,
                       a_pos, a_neg, a_neut)
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    if a_N >= 0:
        del ret[a_N:]
    return ret
