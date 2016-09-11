#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Kiritchenko et al.'s method (2014).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import ENCODING, ESC_CHAR, \
    INFORMATIVE_TAGS, NEGATIVE, POSITIVE, SENT_END_RE, \
    TAB_RE, check_word, normalize

from collections import defaultdict
from math import log
import codecs
import sys


##################################################################
# Constants
FASTMODE = False
POS_IDX = 0
NEG_IDX = 1


##################################################################
# Methods
def _update_stat(a_tok_stat, a_tweet_stat, a_lemmas, a_pos, a_neg):
    """Update statistics on occurrences of words containing seed terms.

    @param a_tok_stat - statistics on term occurrences
    @param a_tweet_stat - statistics on tweets
    @param a_lemmas - lemmas foun in tweet
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded

    @return \c void

    @note modifies `a_tok_stat' and `a_tweet_stat' in place

    """
    if not a_lemmas:
        return
    elif a_lemmas & a_pos:
        a_tweet_stat[POS_IDX] += 1
        for ilemma in a_lemmas:
            a_tok_stat[ilemma][POS_IDX] += 1
    elif a_lemmas & a_neg:
        a_tweet_stat[NEG_IDX] += 1
        for ilemma in a_lemmas:
            a_tok_stat[ilemma][NEG_IDX] += 1
    a_lemmas.clear()


def _read_files(a_stat, a_crp_files, a_pos, a_neg):
    """Read corpus files and populate one-directional co-occurrences.

    @param a_stat - statistics on term occurrences
    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded

    @return 2-tuple - number of positive and number of negative tweets

    @note modifies `a_stat' in place

    """
    print("Reading corpus...", end="", file=sys.stderr)
    i = 0
    itag = ilemma = ""
    tlemmas = set()
    tweet_stat = [0, 0]
    for ifname in a_crp_files:
        with codecs.open(ifname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip().lower()
                if not iline or SENT_END_RE.match(iline) \
                   or iline[0] == ESC_CHAR:
                    if FASTMODE:
                        i += 1
                        if i > 300:
                            break
                    _update_stat(a_stat, tweet_stat, tlemmas, a_pos, a_neg)
                    continue
                try:
                    _, itag, ilemma = TAB_RE.split(iline)
                except:
                    print("Invalid line format at line: {:s}".format(
                        repr(iline)), file=sys.stderr
                    )
                    continue
                ilemma = normalize(ilemma)
                if itag[:2] not in INFORMATIVE_TAGS \
                   or not check_word(ilemma):
                    continue
                tlemmas.add(ilemma)
            _update_stat(a_stat, tweet_stat, tlemmas, a_pos, a_neg)
    print(" done", file=sys.stderr)
    return tweet_stat


def _stat2scores(a_stat, a_n_pos, a_n_neg):
    """Convert statistics to scores.

    @param a_stat - statistics on terms
    @param a_n_pos - total number of positive tweets
    @param a_n_neg - total number of negative tweets

    @return list of terms, their polarities, and scores

    """
    ret = []
    iscore = 0.
    for iterm, (ipos, ineg) in a_stat.iteritems():
        iscore = log(ipos * a_n_neg / (ineg * a_n_pos or 1.), 2)
        ret.append((iterm,
                    POSITIVE if iscore > 0. else NEGATIVE,
                    iscore))
    return ret


def kiritchenko(a_N, a_crp_files, a_pos, a_neg):
    """Method for generating sentiment lexicons using Kiritchenko's approach.

    @param a_N - number of terms to extract
    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded

    @return list of terms sorted according to their polarity scores

    """
    a_pos = set(normalize(w) for w in a_pos)
    a_neg = set(normalize(w) for w in a_neg)

    stat = defaultdict(lambda x: [0, 0])
    n_pos, n_neg = _read_files(stat, a_crp_files, a_pos, a_neg)
    ret = _stat2scores(stat, n_pos, n_neg)
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    return ret
