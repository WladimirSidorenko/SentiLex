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
from germanet import Germanet, normalize
from ising import Ising

from itertools import chain
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize as vecnormalize
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse
import codecs
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
CORPUS_DIR = "corpus_dir"

ESULI = "esuli"
TAKAMURA = "takamura"
W2V = "w2v"

NEGATORS = set(["nicht", "keine", "kein", "keines", "keinem", "keinen"])

# not sure whether "has_hypernym" should be added to SYNRELS
SYNRELS = set(["has_pertainym", "is_related_to", "entails", "is_entailed_by", "has_hyponym", \
                   "has_hypernym"])
ANTIRELS = set(["has_antonym"])

W_DELIM_RE = re.compile('(?:\s|{:s})+'.format('|'.join([re.escape(c) for c in string.punctuation])))
TAB_RE = re.compile(' *\t+ *')
ENCODING = "utf-8"
POSITIVE = "positive"
POS_SET = set()                 # set of positive terms
NEGATIVE = "negative"
NEG_SET = set()                 # set of negative terms
NEUTRAL = "neutral"
NEUT_SET = set()                # set of neutral terms

INFORMATIVE_TAGS = set(["AD", "FM", "NE", "NN", "VV"])
STOP_WORDS = set()
FORM2LEMMA = dict()
lemmatize = lambda x: normalize(x)

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

def _lemmatize(a_form):
    """
    Convert word form to its lemma

    @param a_form - word form for which we should obtain lemma

    @return lemma of the word
    """
    a_form = normalize(a_form)
    if a_form in STOP_WORDS:
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
        lexemes = [lemmatize(iword) for itxt in chain([idef], iexamples) \
                       for iword in W_DELIM_RE.split(itxt)]
        lexemes = [ilex for ilex in lexemes if ilex]
        if lexemes:
            ret[isyn_id] = lexemes
    # create tf/idf vectorizer and appy it to the resulting dictionary
    ivectorizer = TfidfVectorizer(sublinear_tf = True, analyzer = lambda w: w)
    ivectorizer.fit(ret.values())
    ret = {k: _length_normalize(ivectorizer.transform([v])) for k, v in ret.iteritems()}
    return ret

def _lexemes2synset_tfidf(a_germanet, a_synid2tfidf, a_lexemes):
    """
    Convert lexemes to tf/idf vectors corresponding to their synsets

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_lexemes - set of lexemes for which to extract the synsets

    @return set of synset id's which contain lexemes along with their definitions
    """
    ret = set((isyn_id, a_synid2tfidf[isyn_id]) \
                  for ilex in a_lexemes \
                  for isyn_id in a_germanet.lex2synids.get(ilex, []) \
                  if isyn_id in a_synid2tfidf)
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
    Private method for training binary classifiers on synset sets

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
        for ilex_src_id in a_germanet.synid2lex[isrc_id]:
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
                for itrg_id in a_germanet.lex2synids[ilex_trg_id]:
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
            new_set |= a_germanet.synid2lex[isyn_id]
        # print("_es_synid2lex: new_set =", repr(new_set), file = sys.stderr)
        ret.append(new_set)
    return ret

def _es_expand_sets_binary(a_germanet, a_synid2tfidf, a_clf_pos, a_clf_neg, a_pos, a_neg, a_neut):
    """
    Extend sets of polar terms by applying an ensemble of classifiers

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_clf_pos - classifier which predicts the POSITIVE class
    @param a_clf_neg - classifier which predicts the NEGATIVE class
    @param a_pos - set of synset id's and their tf/idf vectors that have positive polarity
    @param a_neg - set of synset id's and their tf/idf vectors that have negative polarity
    @param a_neut - set of synset id's and their tf/idf vectors that have neutral polarity
    @param a_decfunc - decision function for determining polarity of new terms

    @return \c True if sets were changed, \c False otherwise
    """
    ret = False
    pos_candidates = set(); neg_candidates = set(); neut_candidates = set();
    # obtain potential candidates
    _es_generate_candidates(a_germanet, a_synid2tfidf, a_pos, pos_candidates, neg_candidates)
    _es_generate_candidates(a_germanet, a_synid2tfidf, a_neg, neg_candidates, pos_candidates)
    # print("pos_candidates before =", repr(pos_candidates), file = sys.stderr)
    # print("neg_candidates before =", repr(neg_candidates), file = sys.stderr)
    if pos_candidates or neg_candidates:
        ret = True
    else:
        return False
    # remove from potential candidates items that are already in seed sets
    seeds = a_pos | a_neg | a_neut
    pos_candidates -= seeds; neg_candidates -= seeds;
    seeds.clear()
    # print("pos_candidates after =", repr(pos_candidates), file = sys.stderr)
    # print("neg_candidates after =", repr(neg_candidates), file = sys.stderr)
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
    # update positive, negative, and neutral sets
    a_pos |= new_pos; a_neg |= new_neg
    a_neut |= ((pos_candidates | neg_candidates) - (new_pos | new_neg))
    return ret

def _es_expand_sets_ternary(a_germanet, a_synid2tfidf, a_clf, a_pos, a_neg, a_neut):
    """
    Extend sets of polar terms by applying an ensemble of classifiers

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_clf - classifier which makes predictions about the polarity
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity

    @return \c True if sets were changed, \c False otherwise
    """
    ret = False
    pos_candidates = set(); neg_candidates = set(); neut_candidates = set();
    # obtain potential candidates
    _es_generate_candidates(a_germanet, a_synid2tfidf, a_pos, pos_candidates, neg_candidates)
    _es_generate_candidates(a_germanet, a_synid2tfidf, a_neg, neg_candidates, pos_candidates)
    if pos_candidates or neg_candidates:
        ret = True
    else:
        return False
    # remove from potential candidates items that are already in seed sets
    seeds = a_pos | a_neg | a_neut
    pos_candidates -= seeds; neg_candidates -= seeds;
    seeds.clear()
    # obtain predictions for the potential positive terms
    predicted = []
    for iset in (pos_candidates, neg_candidates):
        predicted = a_clf.predict([_es_flatten(iitem[-1]) for iitem in iset])
        for iclass, iitem in zip(predicted, iset):
            if iclass == POSITIVE:
                a_pos.add(iitem)
            elif iclass == NEGATIVE:
                a_neg.add(iitem)
            else:
                a_neut.add(iitem)
    return ret

def esuli_sebastiani(a_germanet, a_N, a_clf_type, a_clf_arity, \
                         a_pos, a_neg, a_neut):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_germanet - GermaNet instance
    @param a_N - number of iterations
    @param a_clf_type - type of classifiers to use (Rocchio or SVM)
    @param a_clf_arity - arity type of classifier (binary or ternary)
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity

    @return \c void
    """
    # obtain Tf/Idf vector for each synset description
    synid2tfidf = _get_tfidf_vec(a_germanet)
    # convert obtained lexemes to synsets
    ipos = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_pos)
    ineg = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_neg)
    ineut = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_neut)
    # train classifier on each of the sets
    changed = True
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
    # iteratively expand sets
    while i < a_N:
        print("Iteration #{:d}".format(i), file = sys.stderr)
        i += 1
        # train classifier on each of the sets and expand these sets afterwards
        if changed:
            if binary_clf:
                 _es_train_binary(clf_pos, clf_neg, ipos, ineg, ineut)
                 changed = _es_expand_sets_binary(a_germanet, synid2tfidf, clf_pos, clf_neg, ipos, ineg, ineut)
            else:
                _es_train_ternary(clf_pos, ipos, ineg, ineut)
                changed = _es_expand_sets_ternary(a_germanet, synid2tfidf, clf_pos, ipos, ineg, ineut)
        # check if sets were changed
        if not changed:
            break
    # lexicalize sets of synset id's
    ipos, ineg, ineut = _es_synid2lex(a_germanet, ipos, ineg, ineut)
    a_pos |= ipos; a_neg |= ineg; a_neut |= ineut;

def _tkm_add_germanet(ising, a_germanet):
    """
    Add lexical nodes from GermaNet ot the Ising spin model

    @param a_ising - instance of the Ising spin model
    @param a_germanet - GermaNet instance

    @return \c void
    """
    # add all lemmas from `FORM2LEMMA` dictionary
    for ilemma in FORM2LEMMA.itervalues():
        ising.add_node(ilemma)
    # add all words from `FORM2LEMMA` dictionary
    for ilemma in a_germanet.lex2synids.iterkeys():
        ising.add_node(ilemma)
    # establish links between synset words and definition lemmas
    def_lexemes = []
    for isynid, (idef, iexamples) in a_germanet.synid2defexmp.iteritems():
        def_lexemes = [lemmatize(iword) for itxt in chain([idef], iexamples) \
                           for iword in W_DELIM_RE.split(itxt)]
        def_lexemes = [ilex for ilex in lexemes if ilex]
        if def_lexemes:
            for idef_lex in def_lexemes:
                for ilex in a_germanet.synid2lex[isynid]:
                    ising.add_edge(ilex, idef_lex)
    # establish links between synset words according to synset relations
    pass

def takamura(a_germanet, a_N, a_corpus_dir, a_pos, a_neg, a_neut):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_germanet - GermaNet instance
    @param a_N - number of terms to extract
    @param a_corpus_dir - directory containing corpus
    @param a_pos - initial set of positive terms to be expanded
    @param a_neg - initial set of negative terms to be expanded
    @param a_neut - initial set of neutral terms to be expanded

    @return \c 0 on success, non-\c 0 otherwise
    """
    # create initial empty network
    ising = Ising()
    # populate network from GermaNet
    _tkm_add_germanet(ising, a_germanet)
    # populate network from corpus
    _tkm_add_corpus(ising, a_corpus_dir)
    # perform MCMC sampling
    # ising.mcmc()

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
    subparser_takamura.add_argument("--form2lemma", "-l", help = "file containing form - lemma correspondances", type = str)
    subparser_takamura.add_argument(GNET_DIR, help = "directory containing GermaNet files")
    subparser_takamura.add_argument(CORPUS_DIR, help = "directory containing raw corpus files")
    subparser_takamura.add_argument("N", help = "final number of additional terms to extract", type = int)
    subparser_takamura.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    subparser_esuli = subparsers.add_parser(ESULI, help = "SentiWordNet model (Esuli and Sebastiani, 2005)")
    subparser_esuli.add_argument("--clf-type", help = "type of classifier to use in ensemble", \
                                     choices = [ROCCHIO, SVM], default = SVM)
    subparser_esuli.add_argument("--clf-arity", help = "classifier's arity", \
                                     choices = [BINARY, TERNARY], default = BINARY)
    subparser_esuli.add_argument("--form2lemma", "-l", help = \
                                     """file containing form - lemma correspondances for words occurring in synset definitions""", \
                                     type = str)
    subparser_esuli.add_argument(GNET_DIR, help = "directory containing GermaNet files")
    subparser_esuli.add_argument("N", help = "number of expansion iterations", type = int)
    subparser_esuli.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    subparser_w2v = subparsers.add_parser(W2V, help = "word2vec model (Mikolov, 2013)")
    subparser_w2v.add_argument("N", help = "final number of terms to extract", type = int)
    subparser_w2v.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    args = argparser.parse_args(a_argv)

    # initialize GermaNet, if needed
    igermanet = None
    if GNET_DIR in args:
        print("Reading GermaNet synsets... ", end = "", file = sys.stderr)
        igermanet = Germanet(getattr(args, GNET_DIR))
        if "form2lemma" in args:
            global lemmatize
            lemmatize = _lemmatize
            _get_form2lemma(args.form2lemma)
        print("done", file = sys.stderr)

    # obtain lists of conjoined terms, if needed

    # read initial seed set
    print("Reading seed sets... ", end = "", file = sys.stderr)
    _read_set(args.seed_set)
    print("done", file = sys.stderr)

    # apply requested method
    print("Expanding seed sets... ", file = sys.stderr)
    if args.dmethod == ESULI:
        esuli_sebastiani(igermanet, args.N, args.clf_type, args.clf_arity, POS_SET, NEG_SET, NEUT_SET)
    elif args.dmethod == TAKAMURA:
        new_sets = takamura(igermanet, args.N, getattr(args, CORPUS_DIR), POS_SET, NEG_SET, NEUT_SET)
    print("Expanding seed sets... done", file = sys.stderr)

    for iclass, iset in ((POSITIVE, POS_SET), (NEGATIVE, NEG_SET), (NEUTRAL, NEUT_SET)):
        for iitem in sorted(iset):
            print((iitem + "\t" + iclass).encode(ENCODING))

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
