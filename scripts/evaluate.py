#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Script for evaluating CRF model.
"""

##################################################################
# Libraries
from alt_argparse import argparser
from alt_fio import AltFileInput, AltFileOutput

import re
import os
import sys
import string

##################################################################
# Constants and Variables
STAT = dict()

SRC_TRG_TAGS = set(["SRC", "BSRC", "TRG", "BTRG"])

# field names for proportional match evaluation
GOLD_SPANS_CNT = 0        # number of spans in gold annotation
PRDCT_SPANS_CNT = 1       # number of spans in automatic annotation
GOLD_INTERSECT_SCORE = 2  # intersection score for gold spans
PRDCT_INTERSECT_SCORE = 3 # intersection score for automatically annotated spans

# field names for exact match
EXACT_MATCH_CNT = GOLD_INTERSECT_SCORE

# flags indicating which span should be updated
GOLD_SPAN = 1
PRDCT_SPAN = 2
ALL_SPANS = GOLD_SPAN | PRDCT_SPAN

#  regular expressions for capturing features and words
PUNCT_RE = re.compile(r"^(?:" + '|'.join([re.escape(c) for c in string.punctuation]) + \
                          "|__COLON__)+$")
FORM_FEAT_RE = re.compile(r"form\[0\]=(\S+)")
CHARCLASS_FEAT_RE = re.compile(r"charclass\[0\]=(\S+)", re.I)
EOS_RE = re.compile(r"(?:\b|^)__EOS__(?:\b|$)")

prev_gld_tag = ""
prev_prdct_tag = ""

gld_span_card = 0
prdct_span_card = 0
gld_intersect_card = 0
prdct_intersect_card = 0

# we need additional set of variables for the case of implicit sentiments
implicit_sentiment = False
exact_match = False
CARD_IDX = 0
INTERSECT_IDX = 1
GOLD_SENTIMENT_CARD = [0, 0]
PRDCT_SENTIMENT_CARD = [0, 0]

##################################################################
# Methods

##################################################################
## non-exact match
# if trg_gld_tag == trg_prdct_tag:
## exact match
# if prev_gld_tag == prev_prdct_tag:
##################################################################
def _span_continues(a_span_type, a_crnt_tag):
    """Check whether span still continues."""
    # determine type of the span to be checked for
    if a_span_type & GOLD_SPAN:
        global GOLD_SENTIMENT_CARD, gld_span_card, gld_intersect_card, prev_gld_tag
        prev_tag = prev_gld_tag
        sentiment_card_stat = GOLD_SENTIMENT_CARD
        span_card, intersect_card = gld_span_card, gld_intersect_card
    elif a_span_type & PRDCT_SPAN:
        global PRDCT_SENTIMENT_CARD, prdct_span_card, prdct_intersect_card, prev_prdct_tag
        prev_tag = prev_prdct_tag
        sentiment_card_stat = PRDCT_SENTIMENT_CARD
        span_card, intersect_card = prdct_span_card, prdct_intersect_card
    else:
        raise Exception("Invalid span type specified")
    prev_trg_tag = _get_trg_tag(prev_tag)
    # check whether previous span still continues
    ret = (prev_tag == a_crnt_tag or _get_trg_tag(prev_tag) == a_crnt_tag)
    # for implicit sentiments, we have to check what the previous and current
    # tags are (for exact match a separate procedure)
    if implicit_sentiment and ret == False:
        crnt_trg_tag = _get_trg_tag(a_crnt_tag)
        reset_counters = False
        # For implicit sentiments, SOURCE and TARGET tags don't terminate the
        # sentiment span -- they only temporary introduce a new span, which
        # however implies that the sentiment span still continues
        if prev_trg_tag == "SENTIMENT" and crnt_trg_tag in SRC_TRG_TAGS:
            sentiment_card_stat[CARD_IDX] = span_card
            sentiment_card_stat[INTERSECT_IDX] = intersect_card
            reset_counters = True
            ret = True
        # If previous span was SOURCE or TARGET, then finalize that span, and
        # add its cardinality to the continued SENTIMENT spans.  If the new
        # span, however, has the tag OTHER, then the SENTIMENT span has to be
        # finalized as well.
        elif prev_tag in SRC_TRG_TAGS:
            # Now we have to distinguish between three cases:
            # - If current tag is also in `SRC_TRG_TAGS', it means that either a SOURCE
            #   span came after a TARGET span or vice versa.  In this case, we can
            #   normally finalize the previous span and add its cardinality to the
            #   continuing SENTIMENT span;
            # - If current tag is a SENTIMENT, we should normally finalize the previous
            #   span, and update SENTIMENT cardinality and intersection cardinality and
            #   continue it;
            # - If current tag is OTHER, we need to finalize both the previous span and
            #   the implied SENTIMENT span.
            sentiment_card_stat[CARD_IDX] += span_card
            sentiment_card_stat[INTERSECT_IDX] += intersect_card
            if crnt_trg_tag == "SENTIMENT" or crnt_trg_tag == "O":
                # finalize the spans here, and simultaneously reset counters to
                # the values of implicit sentiment spans
                finalize_spans(a_span_type)

                if a_span_type & GOLD_SPAN:
                    prev_gld_tag = "SENTIMENT"
                    gld_span_card, gld_intersect_card = sentiment_card_stat
                else:
                    prev_prdct_tag = "SENTIMENT"
                    prdct_span_card, prdct_intersect_card = sentiment_card_stat

                if crnt_trg_tag == "O":
                    # finalize SENTIMENT span if current tag is OTHER.
                    # print >> sys.stderr, "Finalizing implicit sentiment span for", str(a_span_type)
                    finalize_spans(a_span_type)

                # reset implicit cardinality counters
                if a_span_type & GOLD_SPAN:
                    GOLD_SENTIMENT_CARD = [0, 0]
                else:
                    PRDCT_SENTIMENT_CARD = [0, 0]

                ret = True
        # reset cardinality counters if needed
        if reset_counters:
            if a_span_type & GOLD_SPAN:
                gld_span_card = gld_intersect_card = 0
            else:
                prdct_span_card = prdct_intersect_card = 0
    # `ret' indicates whether span still continues or not
    return ret

def _update_stat(a_tag, a_class, a_arg = 1):
    """Update global statistics on tags."""
    global STAT
    if not a_tag:
        return
    if a_tag not in STAT:
        if args.exact:
            STAT[a_tag] = [0, 0, 0]
        else:
            STAT[a_tag] = [0, 0, 0, 0]
    STAT[a_tag][a_class] += a_arg

def _get_trg_tag(a_tag):
    """Get target tag for given tag."""
    if a_tag and a_tag[0] == 'B':
        return a_tag[1:]
    else:
        return a_tag

def _calc_binary_score(a_intersect, a_total):
    """Estimate binary overlapping score.

    @param a_intersect  - number of tokens with matching annotation
    @param a_total      - total number of tokens in span

    @return 1 if at least one token intersects with the span

    """
    return bool(a_total and a_intersect)

def _calc_proportional_score(a_intersect, a_total):
    """Estimate proportional overlapping score.

    @param a_intersect  - number of tokens with matching annotation
    @param a_total      - total number of tokens in span

    @return proportion of tokens with matching annotation

    """
    if a_total == 0:
        return 0
    return float(a_intersect) / a_total

def _calc_exact_score(a_gld_card, a_prdct_card):
    """Estimate binary overlapping score.

    @param a_gld_card   - cardinality of gold span
    @param a_prdct_card - cardinality of predicted span

    @return 1 if both cardinalities are equal, 0 otherwise

    """
    return a_gld_card and a_prdct_card and a_gld_card == a_prdct_card

def _calc_stat(a_gld_tag, a_prdct_tag):
    """Calculate binary/proportional overlap of tagging spans."""
    global STAT, prev_gld_tag, prev_prdct_tag, \
        gld_span_card, prdct_span_card, gld_intersect_card, prdct_intersect_card
    # convert introductory tags (those starting with an `B') to regular ones
    # (by stripping off the initial `B')
    trg_gld_tag = _get_trg_tag(a_gld_tag)
    trg_prdct_tag = _get_trg_tag(a_prdct_tag)
    # print >> sys.stderr, "trg_gld_tag = ", trg_gld_tag
    # print >> sys.stderr, "trg_prdct_tag =", trg_prdct_tag
    # if span introduced by the previous reference tag continues here, we keep
    # incrementing the length of the span
    if gld_span_card and _span_continues(GOLD_SPAN, a_gld_tag):
        gld_span_card += 1
    # otherwise, increment the spans counter
    else:
        # update the statistics for previously ended span
        finalize_spans(GOLD_SPAN)
        gld_span_card = 1

    # we apply the same procedure for the automatically predicted tags
    if prdct_span_card and _span_continues(PRDCT_SPAN, a_prdct_tag):
        prdct_span_card += 1
    # otherwise, increment the counter of the spans
    else:
        finalize_spans(PRDCT_SPAN)
        prdct_span_card = 1

    # check whether the automatically assigned tag and the reference tags are
    # the same and increment intersection counters if they are
    if trg_gld_tag == trg_prdct_tag:
        gld_intersect_card += 1
        prdct_intersect_card += 1
    elif implicit_sentiment:
        if trg_prdct_tag == "SENTIMENT" and trg_gld_tag in SRC_TRG_TAGS:
            prdct_intersect_card += 1
            GOLD_SENTIMENT_CARD[INTERSECT_IDX] += 1
        elif trg_gld_tag == "SENTIMENT" and trg_prdct_tag in SRC_TRG_TAGS:
            gld_intersect_card += 1
            PRDCT_SENTIMENT_CARD[INTERSECT_IDX] += 1
    # remember gold and predicted tags
    prev_gld_tag, prev_prdct_tag = trg_gld_tag, trg_prdct_tag

def _calc_exact_stat(a_gld_tag, a_prdct_tag):
    """Calculate exact overlap of tagging spans."""
    global prev_gld_tag, prev_prdct_tag, gld_span_card, prdct_span_card, \
        gld_intersect_card, prdct_intersect_card
    # convert introductory tags (those starting with an `B') to regular ones
    # (by stripping off the initial `B')
    trg_gld_tag = _get_trg_tag(a_gld_tag)
    trg_prdct_tag = _get_trg_tag(a_prdct_tag)
    # if gold span ended
    if gld_span_card and trg_gld_tag != prev_gld_tag:
        _update_stat(prev_gld_tag, GOLD_SPANS_CNT)
        # new spans have started for both gold and predicted tags
        if prdct_span_card and trg_prdct_tag != prev_prdct_tag:
            _update_stat(prev_prdct_tag, PRDCT_SPANS_CNT)
            # if both previous spans had same tags and equal cardinalities -
            # update exact match counter for previous gold tag
            if prev_gld_tag == prev_prdct_tag:
                _update_stat(prev_gld_tag, EXACT_MATCH_CNT, \
                                 calc_score(gld_span_card, prdct_span_card))
            # reset cardinality counter for predicted span
            prdct_span_card = 0
        # reset cardinality counter for predicted span
        gld_span_card = 0
    # if predicted span ended, but the gold span didn't
    elif prdct_span_card and trg_prdct_tag != prev_prdct_tag:
        _update_stat(prev_prdct_tag, PRDCT_SPANS_CNT)
        # reset cardinality counter for gold span
        prdct_span_card = 0
    # increment counters for gold and predicted spans
    gld_span_card += 1
    prdct_span_card += 1
    # remember gold and predicted tags
    prev_gld_tag, prev_prdct_tag = trg_gld_tag, trg_prdct_tag

def _calc_implicit_exact_stat(a_gld_tag, a_prdct_tag):
    """Calculate exact overlap of tagging spans with implicit sentiments."""
    global prev_gld_tag, prev_prdct_tag, gld_span_card, prdct_span_card

    # convert introductory tags (those starting with an `B') to regular ones
    # (by stripping off the initial `B')
    trg_gld_tag = _get_trg_tag(a_gld_tag)
    prev_trg_gld_tag = _get_trg_tag(prev_gld_tag)
    explicit_gld_ended = False

    trg_prdct_tag = _get_trg_tag(a_prdct_tag)
    prev_trg_prdct_tag = _get_trg_tag(prev_prdct_tag)
    explicit_prdct_ended = False

    # update counters for explicit spans
    if gld_span_card:
        if prev_trg_gld_tag == "SENTIMENT" and trg_gld_tag in SRC_TRG_TAGS:
            GOLD_SENTIMENT_CARD[CARD_IDX] = gld_span_card
            gld_span_card = 0
        elif gld_span_card and prev_trg_gld_tag != a_gld_tag:
            explicit_gld_ended = True
            if prev_trg_gld_tag in SRC_TRG_TAGS:
                GOLD_SENTIMENT_CARD[CARD_IDX] += gld_span_card
    implicit_gld_ended = (explicit_gld_ended and trg_gld_tag == "O")

    if prdct_span_card:
        if prev_trg_prdct_tag == "SENTIMENT" and trg_prdct_tag in SRC_TRG_TAGS:
            PRDCT_SENTIMENT_CARD[CARD_IDX] = prdct_span_card
            prdct_span_card = 0
        elif prev_trg_prdct_tag != a_prdct_tag:
            explicit_prdct_ended = True
            if prev_trg_prdct_tag in SRC_TRG_TAGS:
                PRDCT_SENTIMENT_CARD[CARD_IDX] += prdct_span_card
    implicit_prdct_ended = (explicit_prdct_ended and trg_prdct_tag == "O")

    # finalize explicit sentiment spans
    if explicit_gld_ended:
        _update_stat(prev_trg_gld_tag, GOLD_SPANS_CNT)
        # deal with explicit matching spans (note that the counter of predicted
        # spans will be updated separately)
        if explicit_prdct_ended:
            if prev_trg_gld_tag == prev_trg_prdct_tag:
                _update_stat(prev_trg_gld_tag, EXACT_MATCH_CNT, \
                                 calc_score(gld_span_card, prdct_span_card))
            # now we handle the cases, when one of the spans is explicit
            # SENTIMENT and another one is implicit (cases with two implicit
            # spans will be handled separately)
            elif implicit_gld_ended and implicit_prdct_ended:
                # in case of two ended imlicit spans, if the last tag of one of
                # the spans was SENTIMENT, it will have zero
                if prev_trg_gld_tag == "SENTIMENT" and prev_trg_prdct_tag in SRC_TRG_TAGS:
                    _update_stat("SENTIMENT", EXACT_MATCH_CNT, \
                                     calc_score(gld_span_card, PRDCT_SENTIMENT_CARD[CARD_IDX]))
                elif prev_trg_prdct_tag == "SENTIMENT" and prev_trg_gld_tag in SRC_TRG_TAGS:
                    _update_stat("SENTIMENT", EXACT_MATCH_CNT, \
                                     calc_score(GOLD_SENTIMENT_CARD[CARD_IDX], prdct_span_card))
        # reset the gold span counter to appropriate value
        if trg_gld_tag == "SENTIMENT" and prev_trg_gld_tag in SRC_TRG_TAGS:
            # sentiment tags inherit the cradinality of the previous
            # SOURCE/TARGET spans plus add their own cardinality
            gld_span_card = GOLD_SENTIMENT_CARD[CARD_IDX]
            GOLD_SENTIMENT_CARD[CARD_IDX] = 0
        else:
            gld_span_card = 0

    if explicit_prdct_ended:
        _update_stat(prev_trg_prdct_tag, PRDCT_SPANS_CNT)
        # reset the predicted span counter to appropriate value
        if trg_prdct_tag == "SENTIMENT" and prev_trg_prdct_tag in SRC_TRG_TAGS:
            # sentiment tags inherit the cradinality of the previous
            # SOURCE/TARGET spans plus add their own cardinality
            prdct_span_card = PRDCT_SENTIMENT_CARD[CARD_IDX]
            PRDCT_SENTIMENT_CARD[CARD_IDX] = 0
        else:
            # if SOURCE or TARGET span was followed by another SOURCE or
            # TARGET, add the cardinality of ended span to the cardinality of
            # implied sentiment
            prdct_span_card = 0

    # finalize implicit sentiment spans (only OTHER tag can end this kind of spans)
    if implicit_gld_ended and GOLD_SENTIMENT_CARD[CARD_IDX]:
        _update_stat("SENTIMENT", GOLD_SPANS_CNT)
        if implicit_prdct_ended and PRDCT_SENTIMENT_CARD[CARD_IDX]:
            _update_stat("SENTIMENT", EXACT_MATCH_CNT, \
                             calc_score(GOLD_SENTIMENT_CARD[CARD_IDX], PRDCT_SENTIMENT_CARD[CARD_IDX]))
        GOLD_SENTIMENT_CARD[CARD_IDX] = 0

    if implicit_prdct_ended and PRDCT_SENTIMENT_CARD[CARD_IDX]:
        _update_stat("SENTIMENT", PRDCT_SPANS_CNT)
        PRDCT_SENTIMENT_CARD[CARD_IDX] = 0

    # increment counters for gold and predicted spans
    gld_span_card += 1
    prdct_span_card += 1
    # remember gold and predicted tags
    prev_gld_tag, prev_prdct_tag = trg_gld_tag, trg_prdct_tag

def _finalize_spans(a_span_type = 0):
    """Finalize spans of specific types."""
    # update gold spans
    if a_span_type & GOLD_SPAN:
        global prev_gld_tag, gld_intersect_card, gld_span_card
        if gld_span_card:
            gld_tag = _get_trg_tag(prev_gld_tag)
            _update_stat(gld_tag, GOLD_INTERSECT_SCORE, \
                             calc_score(gld_intersect_card, gld_span_card))
            # remember that new span has started
            _update_stat(gld_tag, GOLD_SPANS_CNT)

            # update counters of implicit spans
            if implicit_sentiment and gld_tag in SRC_TRG_TAGS:
                GOLD_SENTIMENT_CARD[CARD_IDX] += gld_span_card
                GOLD_SENTIMENT_CARD[INTERSECT_IDX] += gld_intersect_card
        gld_span_card = gld_intersect_card = 0

    # update predicted spans
    if a_span_type & PRDCT_SPAN:
        global prev_prdct_tag, prdct_intersect_card, prdct_span_card
        if prdct_span_card:
            prdct_tag = _get_trg_tag(prev_prdct_tag)
            _update_stat(prdct_tag, PRDCT_INTERSECT_SCORE, \
                             calc_score(prdct_intersect_card, prdct_span_card))
            _update_stat(prdct_tag, PRDCT_SPANS_CNT)

            # update counters of implicit spans
            if implicit_sentiment and prdct_tag in SRC_TRG_TAGS:
                PRDCT_SENTIMENT_CARD[CARD_IDX] += prdct_span_card
                PRDCT_SENTIMENT_CARD[INTERSECT_IDX] += prdct_intersect_card
        prdct_span_card = prdct_intersect_card = 0

    # finalize implicit spans at the end of the sequences
    if implicit_sentiment and a_span_type == ALL_SPANS:
        if GOLD_SENTIMENT_CARD[CARD_IDX]:
            _update_stat("SENTIMENT", GOLD_SPANS_CNT)
            _update_stat("SENTIMENT", GOLD_INTERSECT_SCORE, calc_score(*GOLD_SENTIMENT_CARD))
            GOLD_SENTIMENT_CARD[CARD_IDX] = 0
            GOLD_SENTIMENT_CARD[INTERSECT_IDX] = 0

        if PRDCT_SENTIMENT_CARD[CARD_IDX]:
            _update_stat("SENTIMENT", PRDCT_SPANS_CNT)
            _update_stat("SENTIMENT", PRDCT_INTERSECT_SCORE, calc_score(*PRDCT_SENTIMENT_CARD))
            PRDCT_SENTIMENT_CARD[CARD_IDX] = 0
            PRDCT_SENTIMENT_CARD[INTERSECT_IDX] = 0

def _finalize_exact_spans(a_span_type = 0):
    """Calculate final statistics for gold and predicted spans."""
    global prev_gld_tag, prev_prdct_tag, gld_span_card, prdct_span_card
    if gld_span_card and prdct_span_card:
        # update for real and implied gold tags
        gld_tag = _get_trg_tag(prev_gld_tag)
        _update_stat(gld_tag, GOLD_SPANS_CNT)
        # update count of SENTIMENT spans if an implied span has just ended
        # (actually, only the last check in the condition below would suffice,
        # but we included additional checks for safety)
        if implicit_sentiment and gld_tag in SRC_TRG_TAGS:
            _update_stat("SENTIMENT", GOLD_SPANS_CNT)
            GOLD_SENTIMENT_CARD[CARD_IDX] += gld_span_card
        # update counts for real and implied predicted tags
        prdct_tag = _get_trg_tag(prev_prdct_tag)
        _update_stat(prdct_tag, PRDCT_SPANS_CNT)
        if implicit_sentiment and prdct_tag in SRC_TRG_TAGS:
            _update_stat("SENTIMENT", PRDCT_SPANS_CNT)
            PRDCT_SENTIMENT_CARD[CARD_IDX] += prdct_span_card
        # update match counters for explicit spans
        if gld_tag == prdct_tag:
            _update_stat(gld_tag, EXACT_MATCH_CNT, calc_score(gld_span_card, prdct_span_card))
        # update match counters for implied spans if at least one implied
        # counter is present
        if implicit_sentiment and (GOLD_SENTIMENT_CARD[CARD_IDX] or PRDCT_SENTIMENT_CARD[CARD_IDX]):
            # in case when only one counter is present, another counter could
            # be the explcit SENTIMENT span
            if gld_tag == "SENTIMENT":
                GOLD_SENTIMENT_CARD[CARD_IDX] = gld_span_card
            elif prdct_tag == "SENTIMENT":
                PRDCT_SENTIMENT_CARD[CARD_IDX] = prdct_span_card
            # update exact match counter for implicit sentiment spans
            _update_stat("SENTIMENT", EXACT_MATCH_CNT, calc_score(GOLD_SENTIMENT_CARD[CARD_IDX], \
                                                                      PRDCT_SENTIMENT_CARD[CARD_IDX]))
            # reset implicit sentiment counters
            GOLD_SENTIMENT_CARD[CARD_IDX] = PRDCT_SENTIMENT_CARD[CARD_IDX] = 0
    # reset cardinality counters for explicit and implicit spans
    gld_span_card = prdct_span_card = 0

def output_stat(a_foutput):
    """Output tagging statistics for binary/proportional overlap."""
    global STAT
    # output header
    a_foutput.fprint("{:<10s}{:>10s}{:>10s}{:>10s}".format("Tag", "Precision", "Recall", "F-Measure"))
    # output statistics on each tag
    prec = rcall = fscore = 0.0
    gld_intersect = gld_span_card = 0
    prdct_intersect = prdct_span_card = 0
    stat_list = None
    for tag in sorted(STAT.iterkeys()):
        stat_list = STAT[tag]

        gld_intersect = stat_list[GOLD_INTERSECT_SCORE] # number of matched gold spans/span tokens
        gld_span_cnt = stat_list[GOLD_SPANS_CNT] # number of gold spans with given tag
        prdct_intersect = stat_list[PRDCT_INTERSECT_SCORE] # number of matched predicted spans
        prdct_span_cnt = stat_list[PRDCT_SPANS_CNT] # number of predicted spans with given tag

        # print >> sys.stderr, "gld_intersect =", str(gld_intersect)
        # print >> sys.stderr, "gld_span_cnt =", str(gld_span_cnt)
        # print >> sys.stderr, "prdct_intersect =", str(prdct_intersect)
        # print >> sys.stderr, "prdct_span_cnt =", str(prdct_span_cnt)

        prec = float(prdct_intersect) * 100 / prdct_span_cnt if prdct_span_cnt else 0.0
        rcall = float(gld_intersect) * 100 / gld_span_cnt if gld_span_cnt else 0.0
        fscore = 2 * prec * rcall / (prec + rcall) if prec or rcall else 0.0
        a_foutput.fprint("{tag:<10s}{prec:>9.2f}%{rcall:>9.2f}%{fscore:>9.2f}%".format(**locals()))

##################################################################
# Arguments
argparser.description = """Script for evaluating the results of automatic tagging."""
argparser.add_argument("-b", "--binary", help = """consider tagging of two
compared spans as correct if they agree on tagging of at least one word (default,
cf. Breck et al. 2007)""", action = "store_true")
argparser.add_argument("-p", "--proportional", help = """estimate tagging
quality of two compared spans proportionally to the number of overlapping tags (cf.
Johansson and Moschitti, CoNLL 2010)""", action = "store_true")
argparser.add_argument("-x", "--exact", help = """consider tagging of compared spans
as correct only if they completely coincide (cf. Breck et al. 2007)""", action = "store_true")
argparser.add_argument("--implicit-sentiment", help = """SOURCE and TARGET tags always imply
the presence of a SENTIMENT tag""", action = "store_true")
argparser.add_argument("--punct", help = """consider tagging of punctuation marks during
estimation""", action = "store_true")
args = argparser.parse_args()

# check whether SOURCE and TARGET tags always imply a SENTIMENT
implicit_sentiment = args.implicit_sentiment
exact_match = args.exact
# check whether punctuation marks should be skipped
skip_punct = not args.punct
# determine appropriate estimation function
calc_stat = _calc_stat
calc_score = _calc_binary_score
finalize_spans = _finalize_spans

if args.exact:
    if args.implicit_sentiment:
        calc_stat = _calc_implicit_exact_stat
    else:
        calc_stat = _calc_exact_stat
    calc_score = _calc_exact_score
    finalize_spans = _finalize_exact_spans
    # since number of exactly matched sequences is symmetric between gold and
    # predicted spans, they both will refer to the same field
    PRDCT_INTERSECT_SCORE = EXACT_MATCH_CNT
elif args.proportional:
    calc_score = _calc_proportional_score

##################################################################
# Main
foutput = AltFileOutput(encoding = args.encoding, flush = args.flush)
finput = AltFileInput(*args.files, encoding = args.encoding, \
                           print_func = foutput.fprint)

gld_tag = prdct_tag = word = ""
mobj1 = mobj2 = None

for line in finput:
    # skip empty lines
    if not line:
        finalize_spans(ALL_SPANS)
        continue
    # split line in words
    words = line.split()
    # obtain tags
    gld_tag, prdct_tag = words[:2]
    # obtain input word
    if skip_punct:
        assert len(words) > 2, "Unrecognized line format: '{:s}'".format(line)
        mobj1 = FORM_FEAT_RE.search(line)
        mobj2 = CHARCLASS_FEAT_RE.search(line)
        assert mobj2 or mobj1, """`crf_evaluate' could not find the reference word.  The first feature\
 of each word should either be of the form `form[0]=word'\n or you should use the option `--punct'\
 with this script."""
        if mobj1:
            word = mobj1.group(1)
        else:
            word = None
        # prdct_tag, gld_tag, word
        if (mobj2 and mobj2.group(1).lower() == "punct") or (word and PUNCT_RE.match(word)):
            continue
    calc_stat(gld_tag, prdct_tag)
    if EOS_RE.search(line):
        finalize_spans(ALL_SPANS)

# update counters for the last detected spans
finalize_spans(ALL_SPANS)

# print statistics
output_stat(foutput)
