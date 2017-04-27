#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Velikovich's method (2010).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from collections import Counter
from copy import deepcopy
from datetime import datetime
from itertools import chain
from theano import tensor as TT
from sklearn.model_selection import train_test_split
import codecs
import numpy as np
import sys
import theano

from common import BTCH_SIZE, ENCODING, EPSILON, ESC_CHAR, FMAX, FMIN, \
    INFORMATIVE_TAGS, MAX_EPOCHS, MIN_EPOCHS, MIN_TOK_CNT, \
    NEGATIVE_IDX, NEUTRAL_IDX, POSITIVE_IDX, NONMATCH_RE, SENT_END_RE, \
    TAB_RE, check_word, floatX, sgd_updates_adadelta
from common import POSITIVE as POSITIVE_LBL
from common import NEGATIVE as NEGATIVE_LBL
from germanet import normalize


##################################################################
# Constants
DFLT_T = 20
FASTMODE = False
MAX_NGHBRS = 25
TOK_WINDOW = 4                  # it actually corresponds to a window of six
MAX_POS_IDS = 10000

UNK = "%unk%"
UNK_I = 0


##################################################################
# Methods
def _read_files_helper(a_crp_files, a_encoding=ENCODING):
    """Read corpus files and execute specified function.

    @param a_crp_files - files of the original corpus
    @param a_encoding - encoding of the vector file

    @return (Iterator over file lines)

    """
    i = 0
    tokens_seen = False
    for ifname in a_crp_files:
        with codecs.open(ifname, 'r', a_encoding) as ifile:
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
                a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE,
                a_encoding=ENCODING):
    """Read corpus files and populate one-directional co-occurrences.

    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms
    @param a_encoding - encoding of the vector file

    @return (word2vecid, x, y)

    @note constructs statistics in place

    """
    print("Populating corpus statistics...",
          end="", file=sys.stderr)
    word2cnt = Counter(ilemma
                       for _, itag, ilemma in _read_files_helper(a_crp_files,
                                                                 a_encoding)
                       if ilemma is not None and itag[:2] in INFORMATIVE_TAGS
                       and check_word(ilemma))
    print(" done", file=sys.stderr)
    word2vecid = {UNK: UNK_I}
    for w in chain(a_pos, a_neg):
        word2vecid[w] = len(word2vecid)
    # convert words to vector ids if their counters are big enough
    for w, cnt in word2cnt.iteritems():
        if cnt >= MIN_TOK_CNT or a_pos_re.search(w) or a_neg_re.search(w):
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
    label = NEUTRAL_IDX
    for iform, itag, ilemma in _read_files_helper(a_crp_files):
        if ilemma is None:
            if toks:
                max_sent_len = max(max_sent_len, len(toks))
                X.append(deepcopy(toks))
                del toks[:]
                Y.append(label)
                label = NEUTRAL_IDX
            continue
        if ilemma in word2vecid:
            toks.append(word2vecid[ilemma])
        if check_in_seeds(iform, ilemma, a_pos, a_pos_re):
            label = POSITIVE_IDX
        elif check_in_seeds(iform, ilemma, a_neg, a_neg_re):
            label = NEGATIVE_IDX
    X = np.array(
        [x + [UNK_I] * (max_sent_len - len(x))
         for x in X], dtype="int32")
    Y = np.array(Y, dtype="int32")
    return (word2vecid, max_sent_len, X, Y)


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
    # `emb_sum' will be a matrix of size `m x k', where `m' is the mini-batch
    # size and `k' is dimensionality of embeddings
    emb_sum = W[x].sum(axis=1)
    # it actually does not make sense to have an identity matrix in the
    # network, but that's what the original Vo implemenation actually does
    # W2S = theano.shared(value=floatX(np.eye(3)), name="W2S")
    # y_prob = TT.nnet.softmax(TT.dot(W2S, emb_sum.T))
    y_prob = TT.nnet.softmax(emb_sum)
    y_pred = TT.argmax(y_prob, axis=1)

    params = [W]
    cost = -TT.mean(TT.log(y_prob)[TT.arange(y.shape[0]), y])
    updates = sgd_updates_adadelta(params, cost)
    train = theano.function([x, y], cost, updates=updates)

    acc = TT.sum(TT.eq(y, y_pred))
    validate = theano.function([x, y], acc)
    zero_vec = TT.basic.zeros(k)
    zero_out = theano.function([],
                               updates=[(W,
                                         TT.set_subtensor(W[UNK_I, :],
                                                          zero_vec))])
    return (train, validate, zero_out, params)


def vo(a_N, a_crp_files, a_pos, a_neg,
       a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE, a_encoding=ENCODING):
    """Method for generating sentiment lexicons using Velikovich's approach.

    @param a_N - number of terms to extract
    @param a_crp_files - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms
    @param a_encoding - encoding of the vector file

    @return list of terms sorted according to their polarities

    """
    # digitize training set
    word2vecid, max_sent_len, X, Y = _read_files(
        a_crp_files, a_pos, a_neg, a_pos_re, a_neg_re,
        a_encoding
    )
    # initianlize neural net and embedding matrix
    W, k = init_embeddings(len(word2vecid))
    train, validate, zero_out, params = init_nnet(W, k)
    # organize minibatches and run the training
    N = len(Y)
    assert N, "Training set is empty."
    train_idcs, devtest_idcs = train_test_split(
        np.arange(N), test_size=0.1)
    train_N = len(train_idcs)
    devtest_N = float(len(devtest_idcs))
    devtest_x = X[devtest_idcs[:]]
    devtest_y = Y[devtest_idcs[:]]
    btch_size = min(N, BTCH_SIZE)
    epoch_i = 0
    acc = 0
    best_acc = -1
    prev_acc = FMIN
    best_params = []
    while epoch_i < MAX_EPOCHS:
        np.random.shuffle(train_idcs)
        cost = acc = 0.
        start_time = datetime.utcnow()
        for start in np.arange(0, train_N, btch_size):
            end = min(train_N, start + btch_size)
            btch_x = X[train_idcs[start:end]]
            btch_y = Y[train_idcs[start:end]]
            cost += train(btch_x, btch_y)
            zero_out()
        acc = validate(devtest_x, devtest_y) / devtest_N
        if acc >= best_acc:
            best_params = [p.get_value() for p in params]
            best_acc = acc
            sfx = " *"
        else:
            sfx = ''
        end_time = datetime.utcnow()
        tdelta = (end_time - start_time).total_seconds()
        print("Iteration #{:d} ({:.2f} sec): cost = {:.2f}, "
              "accuracy = {:.2%};{:s}".format(epoch_i, tdelta, cost,
                                              acc, sfx),
              file=sys.stderr)
        if abs(prev_acc - acc) < EPSILON and epoch_i > MIN_EPOCHS:
            break
        else:
            prev_acc = acc
        epoch_i += 1
    if best_params:
        for p, val in zip(params, best_params):
            p.set_value(val)
    W = W.get_value()
    ret = []
    for w, w_id in word2vecid.iteritems():
        if w_id == UNK_I:
            continue
        elif w in a_pos or a_pos_re.search(w):
            w_score = FMAX
        elif w in a_neg or a_neg_re.search(w):
            w_score = FMIN
        else:
            w_pol = np.argmax(W[w_id])
            if w_pol == NEUTRAL_IDX:
                continue
            w_score = np.max(W[w_id])
            if (w_pol == POSITIVE_IDX and w_score < 0.) \
               or (w_pol == NEGATIVE_IDX and w_score > 0.):
                w_score *= -1
        ret.append((w,
                    POSITIVE_LBL if w_score > 0. else NEGATIVE_LBL,
                    w_score))
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    if a_N >= 0:
        del ret[a_N:]
    return ret
