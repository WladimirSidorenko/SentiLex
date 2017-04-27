#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module comprising common utilities for lexicon generation.

"""

##################################################################
# Imports
from collections import OrderedDict
from theano import config, tensor as TT
import numpy as np
import re
import sys
import theano

from germanet import normalize
from tokenizer import Tokenizer

##################################################################
# Constants
FMAX = sys.float_info.max
FMIN = -FMAX

ESC_CHAR = ''
SENT_END_RE = re.compile(r"\s*<\s*sentence\s*/\s*>\s*$")
TAB_RE = re.compile(' *\t+ *')
# the `#' and `,' characters were only added later for the NWE and corpus
# methods, when I discovered that hashtags were skipped during processing that
# hastags were skipped from processing
WORD_RE = re.compile(r'^[-#.,\w]+$')
ENCODING = "utf-8"

# not sure whether "has_hypernym" should be added to SYNRELS
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"
INFORMATIVE_TAGS = set(["AD", "FM", "NE", "NN", "VV",
                        "ad", "fm", "ne", "nn", "vv"])
POL2OPPOSITE = {POSITIVE: NEGATIVE, NEGATIVE: POSITIVE}

MIN_TOK_CNT = 4

NEGATORS = set(["nicht", "keine", "kein", "keines", "keinem", "keinen"])
STOP_WORDS = set()
FORM2LEMMA = dict()

ANTONYM = "has_antonym"
SYNONYM = "has_synonym"
ANTIRELS = set([ANTONYM])
# excluded `is_related_to' as it connected `Form' and `unförmig'
SYNRELS = set(["has_participle", "has_pertainym",
               "has_hyponym", "entails", "is_entailed_by"])

NONMATCH_RE = re.compile(r"(?!)")

TOKENIZER = Tokenizer()
MAX_EPOCHS = 500
MIN_EPOCHS = 25
EPSILON = 1e-9

NEGATIVE_IDX = 0
NEUTRAL_IDX = 1
POSITIVE_IDX = 2
BTCH_SIZE = 20


##################################################################
# Imports
def lemmatize(a_form, a_prune=True):
    """
    Convert word form to its lemma

    @param a_form - word form for which we should obtain lemma
    @param a_prune - flag indicating whether uninformative words
                    should be pruned

    @return lemma of the word
    """
    a_form = normalize(a_form)
    if a_prune and a_form in STOP_WORDS:
        return None
    if a_form in FORM2LEMMA:
        return FORM2LEMMA[a_form]
    return a_form


def check_word(a_word):
    """Check if given word forms a valid lexeme

    @param a_word - word to be checked

    @return True if word forms a valid lexeme, \c False otherwise

    """
    return WORD_RE.match(a_word) and all(ord(c) < 256 for c in a_word)


def floatX(a_data, a_dtype=config.floatX):
    """Return numpy array populated with the given data.

    Args:
      data (np.array):
        input tensor
      dtype (class):
        digit type

    Returns:
      np.array:
        array populated with the given data

    """
    return np.asarray(a_data, dtype=a_dtype)


def sgd_updates_adadelta(params, cost, rho=0.95,
                         epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
    """Adadelta update rule.

    Mostly from:
      https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4

    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = floatX(np.zeros_like(param.get_value()))
        exp_sqr_grads[param] = theano.shared(value=empty,
                                             name="exp_grad_%s" % param.name)
        gp = TT.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=empty,
                                           name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * TT.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(TT.sqrt(exp_su + epsilon) / TT.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * TT.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) \
           and (param.name != 'Words'):
            col_norms = TT.sqrt(TT.sum(TT.sqr(stepped_param), axis=0))
            desired_norms = TT.clip(col_norms, 0, TT.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates
