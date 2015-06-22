#!/usr/bin/env python
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
from itertools import chain
from sklearn.svm import LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.feature_extraction.text import TfidfVectorizer

import argparse
import codecs
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

ESULI = "esuli"
TAKAMURA = "takamura"
W2V = "w2v"

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
    # obtain id's of instances pertaining to relevant classes
    pos_ids = set(inst[0] for inst in a_pos)
    neg_ids = set(inst[0] for inst in a_neg)
    neut_ids = set(inst[0] for inst in a_neut)
    pos_train_ids = pos_ids - (neg_ids | neut_ids)
    neg_train_ids = neg_ids - (pos_ids | neut_ids)
    pos_ids.clear(); neg_ids.clear(); neut_ids.clear();
    # create training sets for positive classifier
    instances = []; trg_classes = []
    for syn_id, tfidf_vec in chain(a_pos, a_neg, a_neut):
        instances.append(tfidf_vec)
        trg_classes.append(str(syn_id in pos_train_ids))
    # train positive-vs-all classifier
    a_clf_pos.fit(instances, trg_classes)
    # replace training classes with the negative training instances (assuming
    # that iteration order is the same, we don't modify the actual training
    # vectors)
    del trg_classes[:]
    for syn_id, _ in chain(a_pos, a_neg, a_neut):
        trg_classes.append(str(syn_id in neg_train_ids))
    # train negative-vs-all classifier
    a_clf_neg.fit(instances, trg_classes)

def _es_train_ternary(a_clf, a_pos, a_neg, a_neut):
    """
    Private method for training binary classifiers on synset sets

    @param a_clf - pointer to ternary classifier
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity

    @return \c void
    """
    instances = [inst[-1] for inst in chain(a_pos, a_neg, a_neut)]
    trg_classes = [POSITIVE] * len(a_pos) + [NEGATIVE] * len(a_neg) + NEUTRAL * len(a_neut)
    a_clf.fit(instances, trg_classes)

def _es_generate_candidates(a_candidates, a_class):
    """
    Extend sets of polar terms by applying custom decision function

    @param a_seeds - set of candidate synsets
    @param a_class - class of the items in seed set

    @return 2-tuple containing two sets:
    """
    # for ipos in a_pos:
    #     for isyn_id in a_germanet.lex2synids.get(ipos, []):
    #         for itrg_syn_id, irelname in a_germanet.relations.get(isyn_id, [(None, None)]):
    #             if irelname in SYNRELS:
    #                 for ilex in a_germanet.synid2lex[itrg_syn_id]:
    #                     pos_candidates.add(ilex)
    #             elif irelname in ANTIRELS:
    #                 for ilex in a_germanet.synid2lex[itrg_syn_id]:
    #                     neg_candidates.add(ilex)
    # for ipos in a_pos:
    #     for isyn_id in a_germanet.lex2synids.get(ipos, []):
    #         for itrg_syn_id, irelname in a_germanet.relations.get(isyn_id, [(None, None)]):
    #             if irelname in SYNRELS:
    #                 for ilex in a_germanet.synid2lex[itrg_syn_id]:
    #                     pos_candidates.add(ilex)
    #             elif irelname in ANTIRELS:
    #                 for ilex in a_germanet.synid2lex[itrg_syn_id]:
    #                     neg_candidates.add(ilex)
    # # return the union of three sets
    # return a_pos | a_neg | a_neut

def _es_expand_sets_binary(a_clf_pos, a_clf_neg, a_pos, a_neg, a_neut):
    """
    Extend sets of polar terms by applying an ensemble of classifiers

    @param a_clf_pos - classifier which predicts the POSITIVE class
    @param a_clf_neg - classifier which predicts the NEGATIVE class
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity
    @param a_decfunc - decision function for determining polarity of new terms

    @return \c True if sets were changed, \c False otherwise
    """
    pass

def _es_expand_sets_ternary(a_clf, a_pos, a_neg, a_neut):
    """
    Extend sets of polar terms by applying an ensemble of classifiers

    @param a_clf - classifier which makes predictions about the polarity
    @param a_pos - set of synsets and their tf/idf vectors that have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral polarity
    @param a_decfunc - decision function for determining polarity of new terms

    @return \c True if sets were changed, \c False otherwise
    """
    pass

def esuli_sebastiani(a_germanet, a_N, a_clf_type, a_clf_arity):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_germanet - GermaNet instance
    @param a_N - number of terms to extract
    @param a_clf_type - type of classifiers to use (Rocchio or SVM)
    @param a_clf_arity - arity type of classifier (binary or ternary)

    @return \c 0 on success, non-\c 0 otherwise
    """
    global POS_SET, NEG_SET, NEUT_SET
    # obtain Tf/Idf vector for each synset description
    synid2tfidf = _get_tfidf_vec(a_germanet)
    # convert obtained lexemes to synsets
    ipos = _lexemes2synset_tfidf(a_germanet, synid2tfidf, POS_SET)
    ineg = _lexemes2synset_tfidf(a_germanet, synid2tfidf, NEG_SET)
    ineut = _lexemes2synset_tfidf(a_germanet, synid2tfidf, NEUT_SET)
    # train classifier on each of the sets
    i = 0
    changed = True
    clf_pos = clf_neg = None
    binary_clf = bool(a_clf_arity == BINARY)
    # initialize classifiers
    if a_clf_type == SVM:
        clf_pos = LinearSVC(multiclass = (not binary_clf))
        if binary_clf:
            clf_neg = LinearSVC(multiclass = (not binary_clf))
    elif a_clf_type == ROCCHIO:
        clf_pos = NearestCentroid()
        if binary_clf:
            clf_neg = NearestCentroid()
    else:
        raise RuntimeError("Unknown classifier type: '{:s}'".format(a_clf_type))
    # iteratively expand sets
    while i < a_N:
        # train classifier on each of the sets and expand these sets afterwards
        if changed:
            if binary_clf:
                 _es_train_binary(clf_pos, clf_neg, ipos, ineg, ineut)
                 changed = _es_expand_sets_binary(clf_pos, clf_neg, ipos, ineg, ineut)
            else:
                _es_train_ternary(clf_pos, ipos, ineg, ineut)
                 changed = _es_expand_sets_binary(clf_pos, ipos, ineg, ineut)
        # check if sets were changed
        if not changed:
            break
        i += 1
    return (ipos, ineg, ineut)

def takamura(a_gnet_dir, a_N, a_pos, a_neg, a_neut):
    """
    Method for extending sentiment lexicons using Esuli and Sebastiani method

    @param a_gnet_dir - directory containing GermaNet files
    @param a_N - number of terms to extract
    @param a_pos - initial set of positive terms
    @param a_neg - initial set of negative terms
    @param a_neut - initial set of neutral terms

    @return \c 0 on success, non-\c 0 otherwise
    """
    ret = set()
    return ret

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
    subparser_takamura.add_argument("corpus_dir", help = "directory containing raw corpus files")
    subparser_takamura.add_argument("N", help = "final number of additional terms to extract")
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
    subparser_esuli.add_argument("N", help = "number of expansion iterations")
    subparser_esuli.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    subparser_w2v = subparsers.add_parser(W2V, help = "word2vec model (Mikolov, 2013)")
    subparser_w2v.add_argument("N", help = "final number of terms to extract")
    subparser_w2v.add_argument("seed_set", help = "initial seed set of positive, negative, and neutral terms")

    args = argparser.parse_args(a_argv)

    # initialize GermaNet, if needed
    igermanet = None
    if GNET_DIR in args:
        igermanet = Germanet(getattr(args, GNET_DIR))
        if "form2lemma" in args:
            global lemmatize
            lemmatize = _lemmatize
            _get_form2lemma(args.form2lemma)

    # obtain lists of conjoined terms, if needed

    # read initial seed set
    _read_set(args.seed_set)

    # apply requested method
    if args.dmethod == ESULI:
        new_set = esuli_sebastiani(igermanet, args.N, args.clf_type, args.clf_arity)
    elif args.dmethod == TAKAMURA:
        new_set = takamura(igermanet, args.N, POS_SET, NEG_SET, NEUT_SET)

    for iexpression in sorted(new_set):
        print(iexpression.encode(ENCODING))

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
