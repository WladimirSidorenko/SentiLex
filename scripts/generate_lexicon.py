#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""
Script for generating sentiment lexicons

USAGE:
generate_lexicon.py [OPTIONS] [INPUT_FILES]

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function
from germanet import Germanet, normalize, POS
from ising import Ising, ITEM_IDX, WGHT_IDX, HAS_FXD_WGHT, FXD_WGHT_IDX
from tokenizer import Tokenizer

from itertools import chain, combinations
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as vecnormalize
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse
import codecs
from math import floor, ceil, isnan
import numpy as np
import os
import re
import sys
import string

##################################################################
# Imports
BINARY = "binary"
TERNARY = "ternary"
ROCCHIO = "rocchio"
SVM = "svm"
GNET_DIR = "germanet_dir"
CC_FILE = "cc_file"

VERBOSE = False
ESULI = "esuli"
TAKAMURA = "takamura"
W2V = "w2v"

NEGATORS = set(["nicht", "keine", "kein", "keines", "keinem", "keinen"])

# not sure whether "has_hypernym" should be added to SYNRELS
SYNRELS = set(["has_pertainym", "is_related_to", "entails", "is_entailed_by", "has_hyponym", \
                   "has_hypernym"])
ANTIRELS = set(["has_antonym"])

W_DELIM_RE = re.compile('(?:\s|{:s})+'.format('|'.join([re.escape(c) for c in string.punctuation])))
WORD_RE = re.compile('^[-.\w]+$')
TAB_RE = re.compile(' *\t+ *')
ENCODING = "utf-8"
POSITIVE = "positive"
POS_SET = set()                 # set of positive terms
NEGATIVE = "negative"
NEG_SET = set()                 # set of negative terms
NEUTRAL = "neutral"
NEUT_SET = set()                # set of neutral terms

TOKENIZER = Tokenizer()
INFORMATIVE_TAGS = set(["AD", "FM", "NE", "NN", "VV"])
STOP_WORDS = set()
FORM2LEMMA = dict()
lemmatize = lambda x, a_prune = True: normalize(x)

##################################################################
# Main
def _get_form2lemma(a_fname):
    """
    Read file containing form/lemma correspodences

    @param a_fname - name of input file

    @return \c void (correspondences are read into global variables)
    """
    global STOP_WORDS, FORM2LEMMA

    if not os.path.isfile(a_fname) or not os.access(a_fname, os.R_OK):
        raise RuntimeError("Cannot read from file '{:s}'".format())

    iform = itag = ilemma = ""
    with codecs.open(a_fname, 'r', encoding = ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            iform, itag, ilemma = TAB_RE.split(iline)
            iform = normalize(iform)
            if len(itag) > 1 and itag[:2] in INFORMATIVE_TAGS:
                FORM2LEMMA[iform] = normalize(ilemma)
            else:
                STOP_WORDS.add(iform)

def _read_set(a_fname):
    """
    Read initial seed set of terms.

    @param a_fname - name of input file containing terms

    @return \c void
    """
    global POS_SET, NEG_SET, NEUT_SET
    fields = []
    with codecs.open(a_fname, 'r', encoding = ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            fields = TAB_RE.split(iline)
            if fields[-1] == POSITIVE:
                POS_SET.add(normalize(fields[0]))
            elif fields[-1] == NEGATIVE:
                NEG_SET.add(normalize(fields[0]))
            elif fields[-1] == NEUTRAL:
                NEUT_SET.add(normalize(fields[0]))
            else:
                raise RuntimeError("Unknown field specification: {:s}".format(fields[-1]))

def _lemmatize(a_form, a_prune = True):
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

def _length_normalize(a_vec):
    """
    Length normalize vector

    @param a_vec - vector to be normalized

    @return normalized vector
    """
    return vecnormalize(a_vec)

def _get_tfidf_vec(a_germanet):
    """
    Convert GermaNet synsets as tf/idf vectors of words appearing in their definitions

    @param a_germanet - GermaNet instance

    @return dictionary mapping synset id's to tf/idf vectors
    """
    ret = dict()
    lexemes = []
    # iterate over all synsets
    for isyn_id, (idef, iexamples) in a_germanet.synid2defexmp.iteritems():
        lexemes = [lemmatize(iword) for itxt in chain(idef, iexamples) \
                       for iword in TOKENIZER.tokenize(itxt)]
        lexemes = [ilex for ilex in lexemes if ilex]
        if lexemes:
            ret[isyn_id] = lexemes
    # create tf/idf vectorizer and appy it to the resulting dictionary
    ivectorizer = TfidfVectorizer(sublinear_tf = True, analyzer = lambda w: w)
    ivectorizer.fit(ret.values())
    ret = {k: _length_normalize(ivectorizer.transform([v])) for k, v in ret.iteritems()}
    return ret

def _lexemes2synset_tfidf(a_germanet, a_synid2tfidf, a_lexemes, a_pos = None):
    """
    Convert lexemes to tf/idf vectors corresponding to their synsets

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_lexemes - set of lexemes for which to extract the synsets
    @param a_pos - part-od-speech of lexemes to consider (None for no restriction)

    @return set of synset id's which contain lexemes along with their definitions
    """
    ret = set((isyn_id, a_synid2tfidf[isyn_id]) \
                  for ilex in a_lexemes \
                  for ilexid in a_germanet.lex2lexid.get(ilex, []) \
                  for isyn_id in a_germanet.lexid2synids.get(ilexid, []) \
                  if isyn_id in a_synid2tfidf and (a_pos == None or \
                                                       a_pos == a_germanet.synid2pos[isyn_id]))
    return ret

def _es_flatten(a_smtx):
    """
    Private method for convierting sparse matrices to flat arrays

    @param a_sparse_mtx - sparse matrix to be flattened

    @return flat 1-dimensional array
    """
    return np.squeeze(a_smtx.toarray())

def _es_train_binary(a_clf_pos, a_clf_neg, a_pos, a_neg, a_neut):
    """
    Private method for training binary classifiers on synsets

    @param a_clf_pos - pointer to positive-vs-all classifier
    @param a_clf_pos - pointer to negative-vs-all classifier
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity

    @return \c void
    """
    # obtain id's of instances pertaining to the relevant classes
    pos_ids = set(inst[0] for inst in a_pos)
    neg_ids = set(inst[0] for inst in a_neg)
    neut_ids = set(inst[0] for inst in a_neut)
    pos_train_ids = pos_ids - (neg_ids | neut_ids)
    neg_train_ids = neg_ids - (pos_ids | neut_ids)
    pos_ids.clear(); neg_ids.clear(); neut_ids.clear();
    # create training sets for positive classifier
    instances = []; pos_classes = []; neg_classes = []
    for syn_id, tfidf_vec in chain(a_pos, a_neg, a_neut):
        instances.append(_es_flatten(tfidf_vec))
        pos_classes.append(str(int(syn_id in pos_train_ids)))
        neg_classes.append(str(int(syn_id in neg_train_ids)))
    # train positive-vs-all classifier
    print("Fitting parameters of positive-vs-all model... ", end = "", file = sys.stderr)
    a_clf_pos.fit(instances, pos_classes)
    print("done", file = sys.stderr)
    del pos_classes[:]
    # train negative-vs-all classifier
    print("Fitting parameters of negative-vs-all model... ", end = "", file = sys.stderr)
    a_clf_neg.fit(instances, neg_classes)
    print("done", file = sys.stderr)

def _es_train_ternary(a_clf, a_pos, a_neg, a_neut):
    """
    Private method for training binary classifiers on synset sets

    @param a_clf - pointer to ternary classifier
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity

    @return \c void
    """
    instances = [_es_flatten(inst[-1]) for inst in chain(a_pos, a_neg, a_neut)]
    trg_classes = [POSITIVE] * len(a_pos) + [NEGATIVE] * len(a_neg) + [NEUTRAL] * len(a_neut)
    a_clf.fit(instances, trg_classes)

def _es_generate_candidates(a_germanet, a_synid2tfidf, a_seeds, a_new_same, a_new_opposite):
    """
    Extend sets of polar terms by applying custom decision function

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_seeds - set of candidate synsets
    @param a_new_same - new potential items of the same class
    @param a_new_opposite - new potential items of the opposite class

    @return \c void
    """
    trg_set = None
    for isrc_id, _ in a_seeds:
        # obtain new sets by following links in con rels
        for itrg_id, irelname in a_germanet.con_relations.get(isrc_id, [(None, None)]):
            if irelname in SYNRELS:
                trg_set = a_new_same
            elif irelname in ANTIRELS:
                trg_set = a_new_opposite
            else:
                continue
            if itrg_id in a_synid2tfidf:
                trg_set.add((itrg_id, a_synid2tfidf[itrg_id]))
        # iterate over all lexemes pertaining to this synset
        for ilex_src_id in a_germanet.synid2lexids[isrc_id]:
            # iterate over all target lexemes which the given source lexeme is
            # connected to
            for ilex_trg_id, irelname in a_germanet.lex_relations.get(ilex_src_id, [(None, None)]):
                if irelname in SYNRELS:
                    trg_set = a_new_same
                elif irelname in ANTIRELS:
                    trg_set = a_new_opposite
                else:
                    continue
                # iterate over all synsets which the given target lexeme pertains to
                for itrg_id in a_germanet.lexid2synids[ilex_trg_id]:
                    if itrg_id in a_synid2tfidf:
                        trg_set.add((itrg_id, a_synid2tfidf[itrg_id]))

def _es_synid2lex(a_germanet, *a_sets):
    """
    Convert set of synset id's to corresponding lexemes

    @param a_germanet - GermaNet instance
    @param a_sets - set of synset id's with their tf/idf vectors

    @return set of lexemes corresponding to synset id's
    """
    ret = []
    new_set = None
    for iset in a_sets:
        new_set = set()
        # print("_es_synid2lex: iset =", repr(iset), file = sys.stderr)
        for isyn_id, _ in iset:
            # print("_es_synid2lex: isyn_id =", repr(isyn_id), file = sys.stderr)
            new_set |= set([ilex for ilexid in a_germanet.synid2lexids[isyn_id] \
                                for ilex in a_germanet.lexid2lex[ilexid]])
        # print("_es_synid2lex: new_set =", repr(new_set), file = sys.stderr)
        ret.append(new_set)
    return ret

def _es_expand_sets_binary(a_germanet, a_synid2tfidf, a_clf_pos, a_clf_neg, \
                               a_pos, a_neg, a_neut, a_N):
    """
    Extend sets of polar terms by applying an ensemble of classifiers

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_clf_pos - classifier which predicts the POSITIVE class
    @param a_clf_neg - classifier which predicts the NEGATIVE class
    @param a_pos - set of synset id's and their tf/idf vectors that have positive polarity
    @param a_neg - set of synset id's and their tf/idf vectors that have negative polarity
    @param a_neut - set of synset id's and their tf/idf vectors that have neutral polarity
    @param a_N - target number of terms to extract

    @return \c True if sets were changed, \c False otherwise
    """
    ret = False
    if VERBOSE:
        print("_es_expand_sets_binary: active_synsets = ", a_N - N, file = sys.stderr)
        print("_es_expand_sets_binary: N = ", N, file = sys.stderr)
    pos_candidates = set(); neg_candidates = set();
    # obtain potential candidates
    _es_generate_candidates(a_germanet, a_synid2tfidf, a_pos, pos_candidates, neg_candidates)
    _es_generate_candidates(a_germanet, a_synid2tfidf, a_neg, neg_candidates, pos_candidates)
    # remove from potential candidates items that are already in seed sets
    seeds = a_pos | a_neg | a_neut
    pos_candidates -= seeds; neg_candidates -= seeds;
    seeds.clear()
    if pos_candidates or neg_candidates:
        ret = True
    else:
        if VERBOSE:
            print("_es_expand_sets_binary: no candidates generated", file = sys.stderr)
        return False
    # obtain predictions for the potential positive terms
    if pos_candidates:
        pos_pred = a_clf_pos.predict([_es_flatten(iitem) for _, iitem in pos_candidates])
        neg_pred = a_clf_neg.predict([_es_flatten(iitem) for _, iitem in pos_candidates])
        # obtain new positive terms based on the made predictions
        new_pos = set(iitem for iitem, ipos, ineg in zip(pos_candidates, pos_pred, neg_pred) \
                          if ipos == '1' and ineg != '1')
    else:
        new_pos = set()
    # obtain predictions for the potential negative terms
    if neg_candidates:
        pos_pred = a_clf_pos.predict([_es_flatten(iitem) for _, iitem in neg_candidates])
        neg_pred = a_clf_neg.predict([_es_flatten(iitem) for _, iitem in neg_candidates])
        # obtain new negative terms based on the made predictions
        new_neg = set(iitem for iitem, ipos, ineg in zip(pos_candidates, pos_pred, neg_pred) \
                          if ipos != '1' and ineg == '1')
    else:
        new_neg = set()
    # limit the number of positive and negative terms, if we are
    # reaching the limit
    if VERBOSE:
        print("_es_expand_sets_binary: pos_candidates = ", len(pos_candidates), file = sys.stderr)
        print("_es_expand_sets_binary: new_pos = ", len(new_pos), file = sys.stderr)
        print("_es_expand_sets_binary: neg_candidates = ", len(neg_candidates), file = sys.stderr)
        print("_es_expand_sets_binary: new_neg = ", len(new_neg), file = sys.stderr)
    new_pterms = float(len(new_pos) + len(new_neg))
    if N < new_pterms:
        # calculate proportions of each class
        p_prop = float(len(new_pos)) / new_pterms
        n_prop = int(ceil(N * (1 - p_prop)))
        if VERBOSE:
            print("_es_expand_sets_binary: p_prop =", int(floor(N * p_prop)), file = sys.stderr)
            print("_es_expand_sets_binary: n_prop =", n_prop, file = sys.stderr)
        # update sets according to calculated proportions
        a_pos |= set(list(new_pos)[:int(floor(N * p_prop))])
        a_neg |= set(list(new_neg)[:n_prop])
    else:
        # update positive, negative, and neutral sets
        a_pos |= new_pos; a_neg |= new_neg
    a_neut |= ((pos_candidates | neg_candidates) - (new_pos | new_neg))
    return ret

def _es_expand_sets_ternary(a_germanet, a_synid2tfidf, a_clf, \
                                a_pos, a_neg, a_neut, a_N):
    """
    Extend sets of polar terms by applying an ensemble of classifiers

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_clf - classifier which makes predictions about the polarity
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity
    @param a_N - target number of terms to extract

    @return \c True if sets were changed, \c False otherwise
    """
    ret = False
    pos_candidates = set(); neg_candidates = set();
    # obtain potential candidates
    _es_generate_candidates(a_germanet, a_synid2tfidf, a_pos, pos_candidates, neg_candidates)
    _es_generate_candidates(a_germanet, a_synid2tfidf, a_neg, neg_candidates, pos_candidates)
    # remove from potential candidates items that are already in seed sets
    seeds = a_pos | a_neg | a_neut
    pos_candidates -= seeds; neg_candidates -= seeds;
    seeds.clear()
    if pos_candidates or neg_candidates:
        ret = True
    else:
        return False
    # obtain predictions for potential positive terms
    predicted = []
    i = 0
    for iset in (pos_candidates, neg_candidates):
        predicted = a_clf.predict([_es_flatten(iitem[-1]) for iitem in iset])
        for iclass, iitem in zip(predicted, iset):
            if iclass == POSITIVE:
                a_pos.add(iitem)
                i += 1
            elif iclass == NEGATIVE:
                a_neg.add(iitem)
                i += 1
            else:
                a_neut.add(iitem)
            if i >= N:
                break
        if i >= N:
            break
    return ret

def esuli_sebastiani(a_germanet, a_N, a_clf_type, a_clf_arity, \
                         a_pos, a_neg, a_neut, a_seed_pos = "none"):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_germanet - GermaNet instance
    @param a_N - number of new terms to extract
    @param a_clf_type - type of classifiers to use (Rocchio or SVM)
    @param a_clf_arity - arity type of classifier (binary or ternary)
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no restriction)

    @return \c void
    """
    # obtain Tf/Idf vector for each synset description
    synid2tfidf = _get_tfidf_vec(a_germanet)
    if a_seed_pos == "none":
        a_seed_pos = None
    # convert obtained lexemes to synsets
    ipos = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_pos, a_seed_pos)
    ineg = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_neg, a_seed_pos)
    ineut = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_neut, a_seed_pos)
    # train classifier on each of the sets
    clf_pos = clf_neg = None
    binary_clf = bool(a_clf_arity == BINARY)
    # initialize classifiers
    if a_clf_type == SVM:
        clf_pos = LinearSVC()
        if binary_clf:
            clf_neg = LinearSVC()
    elif a_clf_type == ROCCHIO:
        clf_pos = NearestCentroid()
        if binary_clf:
            clf_neg = NearestCentroid()
    else:
        raise RuntimeError("Unknown classifier type: '{:s}'".format(a_clf_type))
    i = 0
    # iteratively expand sets until we reach the required number of terms
    changed = True
    while changed:
        print("Iteration #{:d}".format(i), file = sys.stderr)
        i += 1
        # train classifier on each of the sets and expand these sets afterwards
        if binary_clf:
            _es_train_binary(clf_pos, clf_neg, ipos, ineg, ineut)
            changed = _es_expand_sets_binary(a_germanet, synid2tfidf, \
                                                 clf_pos, clf_neg, \
                                                 ipos, ineg, ineut, a_N)
        else:
            _es_train_ternary(clf_pos, ipos, ineg, ineut)
            changed = _es_expand_sets_ternary(a_germanet, synid2tfidf, \
                                                  clf_pos, ipos, ineg, ineut, a_N)
            if VERBOSE:
                print("changed = ", changed, file = sys.stderr)
    print("# of polar synsets: ", len(ipos) + len(ineg), file = sys.stderr)
    # lexicalize sets of synset id's
    ipos, ineg, ineut = _es_synid2lex(a_germanet, ipos, ineg, ineut)
    a_pos |= ipos; a_neg |= ineg; a_neut |= ineut;

def _check_word(a_word):
    """
    Check if given word forms a valid lexeme

    @param a_word - word to be checked

    @return \c True if word forms a valid lexeme, \c False otherwise
    """
    return WORD_RE.match(a_word) and all(ord(c) < 256 for c in a_word)

def _tkm_add_germanet(ising, a_germanet):
    """
    Add lexical nodes from GermaNet to the Ising spin model

    @param a_ising - instance of the Ising spin model
    @param a_germanet - GermaNet instance

    @return \c void
    """
    # add all lemmas from the `FORM2LEMMA` dictionary
    for ilemma in FORM2LEMMA.itervalues():
        if ilemma not in STOP_WORDS:
            ising.add_node(ilemma)
    # add all lemmas from synsets
    for ilexid in a_germanet.lexid2synids.iterkeys():
        for ilex in a_germanet.lexid2lex[ilexid]:
            ising.add_node(ilex)
    # establish links between synset words and lemmas appearing in
    # examples and definitions
    def_lexemes = []
    negation_seen = False
    for isynid, (idef, iexamples) in a_germanet.synid2defexmp.iteritems():
        def_lexemes = [lemmatize(iword, a_prune = False) for itxt in chain(idef, iexamples) \
                           for iword in TOKENIZER.tokenize(itxt)]
        def_lexemes = [ilexeme for ilexeme in def_lexemes if ilexeme and \
                           ising.item2nid.get(ilexeme, None) is not None]
        if def_lexemes:
            negation_seen = False
            for idef_lex in def_lexemes:
                if idef_lex in NEGATORS:
                    negation_seen = True
                    continue
                elif idef_lex in STOP_WORDS:
                    continue
                for ilexid in a_germanet.synid2lexids[isynid]:
                    for ilex in a_germanet.lexid2lex[ilexid]:
                        ising.add_edge(ilex, idef_lex, -1. if negation_seen else 1.)
    # establish links between synset lemmas based on the lexical
    # relations
    iwght = 1.
    lemmas1 = lemmas2 = None
    for ifrom, irelset in a_germanet.lex_relations.iteritems():
        lemmas1 = a_germanet.lexid2lex.get(ifrom)
        assert lemmas1 is not None, "No lemma found for id {:s}".format(ifrom)
        for ito, irel in irelset:
            lemmas2 = a_germanet.lexid2lex.get(ito)
            assert lemmas2 is not None, "No lemma found for id {:s}".format(ito)
            if irel in SYNRELS:
                iwght = 1.
            elif irel in ANTIRELS:
                iwght = -1.
            else:
                continue
            for ilemma1 in lemmas1:
                for ilemma2 in lemmas2:
                    ising.add_edge(ilemma1, ilemma2, iwght)
    # establish links between synset lemmas based on the con relations
    for ifrom, irelset in a_germanet.con_relations.iteritems():
        # iterate over all lexemes pertaining to the first synset
        for ilex_id1 in a_germanet.synid2lexids[ifrom]:
            lemmas1 = a_germanet.lexid2lex.get(ilex_id1)
            assert lemmas1 is not None, "No lemma found for id {:s}".format(ifrom)
            # iterate over target synsets and their respective relations
            for ito, irel in irelset:
                if irel in SYNRELS:
                    iwght = 1.
                elif irel in ANTIRELS:
                    iwght = -1.
                else:
                    continue
                # iterate over all lexemes pertaining to the second synset
                for ilex_id2 in a_germanet.synid2lexids[ito]:
                    lemmas2 = a_germanet.lexid2lex.get(ilex_id2)
                    assert lemmas2 is not None, "No lemma found for id {:s}".format(ito)
                    for ilemma1 in lemmas1:
                        for ilemma2 in lemmas2:
                            ising.add_edge(ilemma1, ilemma2, iwght)
    # establish links between lemmas which pertain to the same synset
    ilexemes = set()
    for ilex_ids in a_germanet.synid2lexids.itervalues():
        ilexemes = set([ilex for ilex_id in ilex_ids for ilex in a_germanet.lexid2lex[ilex_id]])
        # generate all possible (n choose 2) combinations of lexemes
        # and put links between them
        for ilemma1, ilemma2 in combinations(ilexemes, 2):
            ising.add_edge(ilemma1, ilemma2, 1.)

def _tkm_add_corpus(ising, a_cc_file):
    """
    Add lexical nodes from corpus to the Ising spin model

    @param a_ising - instance of the Ising spin model
    @param a_cc_file - file containing conjoined word pairs extracted from corpus

    @return \c void
    """
    ifields = []
    iwght = 1.
    ilemma1 = ilemma2 = ""
    with codecs.open(a_cc_file, 'r', ENCODING) as ifile:
        for iline in ifile:
            iline = iline.strip()
            if not iline:
                continue
            ifields = TAB_RE.split(iline)
            if len(ifields) != 3:
                continue
            ilemma1, ilemma2, iwght = ifields
            if _check_word(ilemma1) and _check_word(ilemma2):
                ising.add_edge(normalize(ilemma1), normalize(ilemma2), float(iwght), \
                                   a_add_missing = True)

def takamura(a_germanet, a_N, a_cc_file, a_pos, a_neg, a_neut, a_plot = None):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_germanet - GermaNet instance
    @param a_N - number of terms to extract
    @param a_cc_file - file containing coordinatively conjoined phrases
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded
    @param a_plot - name of file in which generated statics plots should be
                    saved (None if no plot should be generated)

    @return \c 0 on success, non-\c 0 otherwise
    """
    # estimate the number of terms to extract
    seed_set = a_pos | a_neg
    a_N -= len(seed_set)
    if a_N < 1:
        return
    # create initial empty network
    ising = Ising()
    # populate network from GermaNet
    print("Adding GermaNet synsets...", end = "", file = sys.stderr)
    _tkm_add_germanet(ising, a_germanet)
    print("done (Ising model has {:d} nodes)".format(ising.n_nodes), file = sys.stderr)
    # populate network from corpus
    print("Adding coordinate phrases from corpus...", end = "", file = sys.stderr)
    _tkm_add_corpus(ising, a_cc_file)
    print("done (Ising model has {:d} nodes)".format(ising.n_nodes), file = sys.stderr)
    # reweight edges
    ising.reweight()
    # set fixed weights for words pertaining to the positive, negative, and neutral set
    for ipos in a_pos:
        if ipos in ising:
            ising[ipos][FXD_WGHT_IDX] = 1.
        else:
            ising.add_node(ipos, 1.)
        ising[ipos][HAS_FXD_WGHT] = 1
    for ineg in a_neg:
        if ineg in ising:
            ising[ineg][FXD_WGHT_IDX] = -1.
        else:
            ising.add_node(ineg, -1.)
        ising[ineg][HAS_FXD_WGHT] = 1
    for ineut in a_neut:
        if ineut in ising:
            ising[ineut][FXD_WGHT_IDX] = 0.
        else:
            ising.add_node(ineut, 0.)
        ising[ineut][HAS_FXD_WGHT] = 1
    ising.train(a_plot = a_plot)
    # nodes = [inode[ITEM_IDX] for inode in sorted(ising.nodes, key = lambda x: x[WGHT_IDX]) \
    #              if inode[ITEM_IDX] not in seed_set]
    seed_set |= a_neut
    nodes = [inode for inode in sorted(ising.nodes, key = lambda x: abs(x[WGHT_IDX]), reverse = True) \
                 if inode[ITEM_IDX] not in seed_set]
    seed_set.clear()
    # populate polarity sets and flush all terms to an external file
    i = 0
    with open(os.path.join("data", "ising_full.txt"), 'w') as ofile:
        for inode in nodes:
            if type(inode[WGHT_IDX]) != float:
                print(inode[ITEM_IDX].encode(ENCODING), "\t", inode[WGHT_IDX], file = ofile)
            else:
                if i < a_N:
                    if inode[WGHT_IDX] > 0:
                        a_pos.add(inode[ITEM_IDX])
                    elif inode[WGHT_IDX] < 0:
                        a_neg.add(inode[ITEM_IDX])
                    else:
                        i -= 1
                    i += 1
                print(inode[ITEM_IDX].encode(ENCODING), "\t{:f}".format(nodes[WGHT_IDX]), file = ofile)

def main(a_argv):
    """
    Main method for generating sentiment lexicons

    @param a_argv - command-line arguments

    @return \c 0 on success, non-\c 0 otherwise
    """
    argparser = argparse.ArgumentParser(description = """Script for \
generating sentiment lexicons.""")
    # add type-specific subparsers
    subparsers = argparser.add_subparsers(help = "lexicon expansion method to use", dest = "dmethod")

    subparser_takamura = subparsers.add_parser(TAKAMURA, help = "Ising spin model (Takamura, 2005)")
    subparser_takamura.add_argument("--form2lemma", "-l", help = "file containing form - lemma correspondences", type = str)
    subparser_takamura.add_argument("--plot", "-p", \
                                        help = "suffix of files in which to store the plot image", \
                                        type = str, default = "")
    subparser_takamura.add_argument(GNET_DIR, help = "directory containing GermaNet files")
    subparser_takamura.add_argument(CC_FILE, help = "file containing coordinatively conjoined phrases")
    subparser_takamura.add_argument("N", help = "final number of additional terms to extract", type = int)
    subparser_takamura.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    subparser_esuli = subparsers.add_parser(ESULI, help = "SentiWordNet model (Esuli and Sebastiani, 2005)")
    subparser_esuli.add_argument("--clf-type", help = "type of classifier to use in ensemble", \
                                     choices = [ROCCHIO, SVM], default = SVM)
    subparser_esuli.add_argument("--clf-arity", help = "classifier's arity", \
                                     choices = [BINARY, TERNARY], default = BINARY)
    subparser_esuli.add_argument("--seed-pos", help = "part-of-speech of seed synsets (\"none\" for no restriction)", \
                                     choices = ["none"] + [p[:-1] for p in POS], default = "none")
    subparser_esuli.add_argument("--form2lemma", "-l", help = \
                                     """file containing form - lemma correspondences for words occurring in synset definitions""", \
                                     type = str)
    subparser_esuli.add_argument("N", help = "number of expansion iterations", type = int)
    subparser_esuli.add_argument(GNET_DIR, help = "directory containing GermaNet files")
    subparser_esuli.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    # disabled.  look at the C++ implementation instead.
    # subparser_w2v = subparsers.add_parser(W2V, help = "word2vec model (Mikolov, 2013)")
    # subparser_w2v.add_argument("N", help = "final number of terms to extract", type = int)
    # subparser_w2v.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    args = argparser.parse_args(a_argv)

    # initialize GermaNet, if needed
    igermanet = None
    if GNET_DIR in args:
        print("Reading GermaNet synsets... ", end = "", file = sys.stderr)
        igermanet = Germanet(getattr(args, GNET_DIR))
        if "form2lemma" in args and args.form2lemma is not None:
            global lemmatize
            lemmatize = _lemmatize
            _get_form2lemma(args.form2lemma)
        print("done", file = sys.stderr)

    # obtain lists of conjoined terms, if needed

    # read initial seed set
    print("Reading seed sets... ", end = "", file = sys.stderr)
    _read_set(args.seed_set)
    print("done", file = sys.stderr)

    N = args.N - (len(a_pos) + len(a_neg))
    # only perform expansion if the number of seed terms is less than
    # the request number of polar items
    if N > 1:
        # apply requested method
        print("Expanding seed sets... ", file = sys.stderr)
        if args.dmethod == ESULI:
            esuli_sebastiani(igermanet, args.N, args.clf_type, args.clf_arity, \
                                 POS_SET, NEG_SET, NEUT_SET, args.seed_pos)
        elif args.dmethod == TAKAMURA:
            takamura(igermanet, args.N, getattr(args, CC_FILE), POS_SET, NEG_SET, NEUT_SET, \
                         a_plot = args.plot or None)
        print("Expanding seed sets... done", file = sys.stderr)

    for iclass, iset in ((POSITIVE, POS_SET), (NEGATIVE, NEG_SET), (NEUTRAL, NEUT_SET)):
        for iitem in sorted(iset):
            print((iitem + "\t" + iclass).encode(ENCODING))

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
