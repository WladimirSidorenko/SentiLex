#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating lexicon using Velikovich's method (2010).

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from datetime import datetime
from lasagne.init import HeUniform, Orthogonal
from sklearn.model_selection import train_test_split
from theano import tensor as TT
import codecs
import numpy as np
import re
import sys
import theano

from common import ENCODING, EPSILON, FMAX, FMIN, MAX_EPOCHS, MIN_EPOCHS, \
    NONMATCH_RE, NEGATIVE_IDX, NEUTRAL_IDX, POSITIVE_IDX, \
    floatX, sgd_updates_adadelta
from common import POSITIVE as POSITIVE_LBL
from common import NEGATIVE as NEGATIVE_LBL
from germanet import normalize


##################################################################
# Constants
SPACE_RE = re.compile(r"\s+")
ORTHOGONAL = Orthogonal()
HE_UNIFORM = HeUniform()


##################################################################
# Methods
def digitize_trainset(w2i, a_pos, a_neg, a_neut, a_pos_re, a_neg_re):
    """Method for generating sentiment lexicons using Velikovich's approach.

    @param a_N - number of terms to extract
    @param a_emb_fname - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms

    @return list of terms sorted according to their polarities

    """
    X = []
    Y = []

    def add_seeds(seeds, label):
        for iterm in seeds:
            iterm = normalize(iterm)
            if iterm in w2i:
                X.append(w2i[iterm])
                Y.append(label)

    add_seeds(a_pos, POSITIVE_IDX)
    add_seeds(a_neg, NEGATIVE_IDX)
    add_seeds(a_neut, NEUTRAL_IDX)

    for iterm, idx in w2i.iteritems():
        if a_pos_re.match(iterm):
            X.append(idx)
            Y.append(POSITIVE_IDX)
        elif a_neg_re.match(iterm):
            X.append(idx)
            Y.append(NEGATIVE_IDX)
    return (np.asarray(X, dtype="int32"),
            np.asarray(Y, dtype="int32"))


def read_embeddings(fname, encoding):
    """Read embeddings from file and populate an mebedding matrix.

    Args:
      fname (str): name of the embedding file
      encoding (str): file encoding

    Returns:
      3-tuple(dict, theano.shared, int): embedding matrix, vector
        dimensionality

    """
    i = 0
    w2i = {}
    EMBS = None
    with codecs.open(fname, 'r', encoding=encoding, errors="replace") as ifile:
        for iline in ifile:
            iline = iline.strip()
            if EMBS is None:
                nterms, ndim = [int(t) for t in SPACE_RE.split(iline)]
                EMBS = floatX(np.empty((nterms, ndim)))
                continue
            fields = SPACE_RE.split(iline)
            try:
                EMBS[i] = np.array([float(f) for f in fields[-ndim:]])
                w2i[' '.join(fields[:-ndim])] = i
            except:
                print("Invalid line format: {!r}".format(iline),
                      file=sys.stderr)
                continue
            i += 1
    EMBS = theano.shared(value=EMBS, name="EMBS")
    return (w2i, EMBS, ndim)


def init_nnet(W, n_classes, vec_dim):
    """Initialize neural network.

    Args:
      W (theano.shared): embedding matrix
      n_classes: number of classes to be predicted
      vec_dim: dimensionality of the embeddings

    """
    w_idx = TT.iscalar(name="w_idx")
    y_gold = TT.iscalar(name="y_gold")
    embs = W[w_idx]
    Theta = theano.shared(value=ORTHOGONAL.sample((n_classes, vec_dim)),
                          name="Theta")
    beta = theano.shared(value=HE_UNIFORM.sample((1, n_classes)), name="beta")
    y_probs = TT.nnet.softmax(TT.dot(Theta, embs.T).flatten() + beta).flatten()
    params = [Theta]
    cost = -TT.mean(TT.log(y_probs[y_gold]))
    updates = sgd_updates_adadelta(params, cost)
    train = theano.function([w_idx, y_gold], cost, updates=updates)
    y_pred = TT.argmax(y_probs)
    y_score = y_probs[y_pred]
    predict = theano.function([w_idx], (y_pred, y_score))
    acc = TT.eq(y_gold, y_pred)
    validate = theano.function([w_idx, y_gold], acc)
    return (train, validate, predict, params)


def tang(a_N, a_emb_fname, a_pos, a_neg, a_neut,
         a_pos_re=NONMATCH_RE, a_neg_re=NONMATCH_RE,
         a_encoding=ENCODING):
    """Method for generating sentiment lexicons using Velikovich's approach.

    @param a_N - number of terms to extract
    @param a_emb_fname - files of the original corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded
    @param a_pos_re - regular expression for matching positive terms
    @param a_neg_re - regular expression for matching negative terms
    @param a_neg_re - regular expression for matching negative terms
    @param a_encoding - encoding of the vector file

    @return list of terms sorted according to their polarities

    """
    w2i, EMBS, ndim = read_embeddings(a_emb_fname, a_encoding)
    X, Y = digitize_trainset(w2i, a_pos, a_neg, a_neut,
                             a_pos_re, a_neg_re)
    train, validate, predict, params = init_nnet(EMBS,
                                                 len(set(Y)), ndim)
    best_params = []
    best_acc = acc = -1
    N = len(Y)
    train_idcs, devtest_idcs = train_test_split(np.arange(N),
                                                test_size=0.1)
    devtest_N = float(len(devtest_idcs))
    devtest_X = X[devtest_idcs]
    devtest_Y = Y[devtest_idcs]
    # train
    epoch_i = 0
    prev_cost = 0
    while epoch_i < MAX_EPOCHS:
        np.random.shuffle(train_idcs)
        cost = 0.
        start_time = datetime.utcnow()
        for idx in train_idcs:
            x_i, y_i = X[idx], Y[idx]
            cost += train(x_i, y_i)
        acc = 0.
        for x_i, y_i in zip(devtest_X, devtest_Y):
            acc += validate(x_i, y_i)
        acc /= devtest_N
        if acc >= best_acc:
            best_params = [p.get_value() for p in params]
            best_acc = acc
            sfx = " *"
        else:
            sfx = ''
        end_time = datetime.utcnow()
        tdelta = (end_time - start_time).total_seconds()
        print("Iteration #{:d} ({:.2f} sec): cost = {:.2f}, "
              "accuracy = {:.2%};{:s}".format(epoch_i, tdelta,
                                              cost, acc, sfx),
              file=sys.stderr)
        if abs(prev_cost - cost) < EPSILON and epoch_i > MIN_EPOCHS:
            break
        else:
            prev_cost = cost
        epoch_i += 1
    if best_params:
        for p, val in zip(params, best_params):
            p.set_value(val)
    # apply trained classifier to unseen data
    ret = []
    for w, w_idx in w2i.iteritems():
        if normalize(w) in a_pos or a_pos_re.match(w):
            pol_cls = POSITIVE_LBL
            pol_score = FMAX
        elif normalize(w) in a_neg or a_neg_re.match(w):
            pol_cls = NEGATIVE_LBL
            pol_score = FMIN
        else:
            pol_idx, pol_score = predict(w_idx)
            pol_score = pol_score.item(0)
            if pol_idx == POSITIVE_IDX:
                pol_cls = POSITIVE_LBL
            elif pol_idx == NEGATIVE_IDX:
                pol_cls = NEGATIVE_LBL
            else:
                continue
            ret.append((w, pol_cls, pol_score))
    ret.sort(key=lambda el: abs(el[-1]), reverse=True)
    return ret
