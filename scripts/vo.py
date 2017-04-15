#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Velikovich's method (2010).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from collections import Counter, OrderedDict
from copy import deepcopy
from theano import config, tensor as TT
from datetime import datetime
import codecs
import numpy as np
import sys
import theano

from common import ENCODING, ESC_CHAR, INFORMATIVE_TAGS, \
    MIN_TOK_CNT, NONMATCH_RE, SENT_END_RE, TAB_RE, check_word
from germanet import normalize


##################################################################
# Constants
DFLT_T = 20
FASTMODE = False
MAX_NGHBRS = 25
TOK_WINDOW = 4                  # it actually corresponds to a window of six
MAX_POS_IDS = 10000

NEGATIVE = 0
NEUTRAL = 1
POSITIVE = 2

MAX_EPOCHS = 5
BTCH_SIZE = 20
UNK = "%unk%"
UNK_I = 0


##################################################################
# Methods
def _read_files_helper(a_crp_files):
    """Read corpus files and execute specified function.

    @param a_crp_files - files of the original corpus

    @return (Iterator over file lines)

    """
    i = 0
    tokens_seen = False
    for ifname in a_crp_files:
        with codecs.open(ifname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip().lower()
                if not iline or SENT_END_RE.match(iline):
                    continue
                elif iline[0] == ESC_CHAR:
                    if FASTMODE:
                        i += 1
                        if i > 300:
                            break
                    if tokens_seen:
                        tokens_seen = False
                        yield None, None, None
                    continue
                try:
                    iform, itag, ilemma = TAB_RE.split(iline)
                except:
                    print("Invalid line format at line: {:s}".format(
                        repr(iline)), file=sys.stderr
                    )
                    continue
                tokens_seen = True
                yield iform, itag, normalize(ilemma)
        yield None, None, None


def _read_files(a_crp_files, a_pos, a_neg,
                a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE):
    """Read corpus files and populate one-directional co-occurrences.

    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return (word2vecid, x, y)

    @note constructs statistics in place

    """
    print("Populating corpus statistics...",
          end="", file=sys.stderr)
    word2cnt = Counter(ilemma
                       for _, itag, ilemma in _read_files_helper(a_crp_files)
                       if ilemma is not None and itag[:2] in INFORMATIVE_TAGS
                       and check_word(ilemma))
    print(" done", file=sys.stderr)
    word2vecid = {UNK: UNK_I}
    # convert words to vector ids if their counters are big enough
    for w, cnt in word2cnt.iteritems():
        if cnt >= MIN_TOK_CNT or w in a_pos or w in a_neg:
            word2vecid[w] = len(word2vecid)
    word2cnt.clear()

    # generate the training set
    def check_in_seeds(a_form, a_lemma, a_seeds, a_seed_re):
        if a_seed_re.search(a_form) or a_seed_re.search(a_lemma) \
           or a_form in a_seeds or normalize(a_form) in a_seeds \
           or a_lemma in a_seeds:
            return True
        return False

    max_sent_len = 0
    X = []
    Y = []
    toks = []
    label = NEUTRAL
    for iform, itag, ilemma in _read_files_helper(a_crp_files):
        if ilemma is None:
            if toks:
                max_sent_len = max(max_sent_len, len(toks))
                X.append(deepcopy(toks))
                del toks[:]
                Y.append(label)
                label = NEUTRAL
            continue
        if ilemma in word2vecid:
            toks.append(word2vecid[ilemma])
        if check_in_seeds(iform, ilemma, a_pos, a_pos_re):
            label = POSITIVE
        elif check_in_seeds(iform, ilemma, a_neg, a_neg_re):
            label = NEGATIVE
    X = np.array(
        [x + [UNK_I] * (max_sent_len - len(x))
         for x in X], dtype="int32")
    Y = np.array(Y, dtype="int32")
    return (word2vecid, max_sent_len, X, Y)


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


def init_embeddings(vocab_size, k=3):
    """Uniformly initialze lexicon scores for each vocabulary word.

    Args:
      vocab_size (int): vocabulary size
      k (int): dimensionality of embeddings

    Returns:
      2-tuple(theano.shared, int): embedding matrix, vector dimmensionality

    """
    rand_vec = np.random.uniform(-0.25, 0.25, k)
    W = floatX(np.broadcast_to(rand_vec,
                               (vocab_size, k)))
    # zero-out the vector of unknown terms
    W[UNK_I] *= 0.
    return theano.shared(value=W, name='W'), k


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


def init_nnet(W, k):
    """Initialize neural network.

    Args:
      W (theano.shared): embedding matrix
      k: dimensionality of the vector

    """
    # `x' will be a matrix of size `m x n', where `m' is the mini-batch size
    # and `n' is the maximum observed sentence length times the dimensionality
    # of embeddings (`k')
    x = TT.imatrix(name='x')
    # `y' will be a vectors of size `m', where `m' is the mini-batch size
    y = TT.ivector(name='y')
    params = [W]
    # `emb_sum' will be a matrix of size `m x k', where `m' is the mini-batch
    # size and `k' is dimensionality of embeddings
    emb_sum = W[x].sum(axis=1)
    # it actually does not make sense to have an identity matrix in the
    # network, but that's what the original Vo implemenation actually does
    # W2S = theano.shared(value=floatX(np.eye(3)), name="W2S")
    # y_prob = TT.nnet.softmax(TT.dot(W2S, emb_sum.T))
    y_prob = TT.nnet.softmax(emb_sum)
    y_pred = TT.argmax(y_prob, axis=1)

    cost = -TT.mean(TT.log(y_prob)[TT.arange(y.shape[0]), y])
    acc = TT.sum(TT.eq(y, y_pred))

    updates = sgd_updates_adadelta(params, cost)
    train = theano.function([x, y], cost, updates=updates)
    validate = theano.function([x, y], acc)
    zero_vec = TT.basic.zeros(k)
    zero_out = theano.function([],
                               updates=[(W,
                                         TT.set_subtensor(W[UNK_I, :],
                                                          zero_vec))])
    return (train, validate, zero_out)


def vo(a_N, a_crp_files, a_pos, a_neg,
       a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE):
    """Method for generating sentiment lexicons using Velikovich's approach.

    @param a_N - number of terms to extract
    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return list of terms sorted according to their polarities

    """
    # digitize training set
    word2vecid, max_sent_len, X, Y = _read_files(
        a_crp_files, a_pos, a_neg, a_pos_re, a_neg_re
    )
    # initianlize neural net and embedding matrix
    W, k = init_embeddings(len(word2vecid))
    train, validate, zero_out = init_nnet(W, k)
    # organize minibatches and run the training
    N = len(Y)
    assert N, "Training set is empty."
    idcs = np.arange(N)
    btch_size = min(N, BTCH_SIZE)
    epoch_i = 0
    while epoch_i < MAX_EPOCHS:
        np.random.shuffle(idcs)
        cost = acc = 0.
        start_time = datetime.utcnow()
        for start in np.arange(0, N, btch_size):
            end = min(N, start + btch_size)
            btch_x = X[idcs[start:end]]
            btch_y = Y[idcs[start:end]]
            cost += train(btch_x, btch_y)
            zero_out()
            acc += validate(btch_x, btch_y)
        end_time = datetime.utcnow()
        tdelta = (end_time - start_time).total_seconds()
        print("Iteration #{:d} ({:.2f} sec): cost = {:.2f}, "
              "accuracy = {:.2%};".format(epoch_i, tdelta,
                                          cost, acc / float(N)))
        epoch_i += 1
