#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

"""Module for generating sentiment lexicon using Esuli/Sebstiani.

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from common import lemmatize, ANTIRELS, SYNRELS, TOKENIZER, \
    POSITIVE, NEGATIVE, NEUTRAL

from collections import defaultdict
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors.nearest_centroid import NearestCentroid, \
    check_is_fitted, check_array, pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize as vecnormalize
from sklearn.svm import LinearSVC

import numpy as np
import sys

##################################################################
# Constants
ITERS = (0, 2, 4, 6)


##################################################################
# Class
class Rocchio(NearestCentroid):
    """Subclass of NearestCentroid with ``predict_proba``.

    """

    def predict_proba(self, X):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Args:
          X (array-like, shape = [n_samples, n_features]): input instamce

        Returns:
          C (array, shape = [n_samples]):

        Notes:
          If the metric constructor parameter is "precomputed", X is assumed to
          be the distance matrix between the data to be predicted and
          ``self.centroids_``.

        """
        check_is_fitted(self, 'centroids_')
        X = check_array(X, accept_sparse='csr')
        y = pairwise_distances(
            X, self.centroids_, metric=self.metric
        )
        y = np.sum(y) - y
        return y / np.sum(y)


##################################################################
# Methods
def softmax(a_arr):
    """Compute softmax of array.

    @param a_arr - input array

    @return softmax of the array

    """
    exp = np.exp(a_arr)
    return exp / np.sum(exp)


def _flatten(a_smtx):
    """
    Private method for convierting sparse matrices to flat arrays

    @param a_sparse_mtx - sparse matrix to be flattened

    @return flat 1-dimensional array
    """
    return np.squeeze(a_smtx.toarray())


def _length_normalize(a_vec):
    """Length normalize vector

    @param a_vec - vector to be normalized

    @return normalized vector

    """
    return vecnormalize(a_vec)


def _get_tfidf_vec(a_germanet):
    """Convert GermaNet synsets to tf/idf vectors of glosses.

    @param a_germanet - GermaNet instance

    @return dictionary mapping synset id's to tf/idf vectors

    """
    ret = dict()
    lexemes = []
    # iterate over all synsets
    for isyn_id, (idef, iexamples) in a_germanet.synid2defexmp.iteritems():
        lexemes = [lemmatize(iword) for itxt in chain(idef, iexamples)
                   for iword in TOKENIZER.tokenize(itxt)]
        lexemes = [ilex for ilex in lexemes if ilex]
        if lexemes:
            ret[isyn_id] = lexemes
    # create tf/idf vectorizer and appy it to the resulting dictionary
    ivectorizer = TfidfVectorizer(sublinear_tf=True, analyzer=lambda w: w)
    ivectorizer.fit(ret.values())
    ret = {k: _length_normalize(ivectorizer.transform([v]))
           for k, v in ret.iteritems()}
    return ret


def _lexemes2synset_tfidf(a_germanet, a_synid2tfidf, a_lexemes, a_pos=None):
    """Convert lexemes to tf/idf vectors corresponding to their synsets

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_lexemes - set of lexemes for which to extract the synsets
    @param a_pos - part-od-speech of lexemes to consider
      (None for no restriction)

    @return set of synset id's with definitions which contain lexemes

    """
    ret = set((isyn_id, a_synid2tfidf[isyn_id])
              for ilex in a_lexemes
              for ilexid in a_germanet.lex2lexid.get(ilex, [])
              for isyn_id in a_germanet.lexid2synids.get(ilexid, [])
              if isyn_id in a_synid2tfidf and (a_pos is None or
                                               a_pos ==
                                               a_germanet.synid2pos[isyn_id]))
    return ret


def _synid2lex(a_germanet, *a_sets):
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
        # print("_synid2lex: iset =", repr(iset), file = sys.stderr)
        for isyn_id, _ in iset:
            # print("_synid2lex: isyn_id =", repr(isyn_id), file=sys.stderr)
            new_set |= set([ilex for ilexid in a_germanet.synid2lexids[isyn_id]
                            for ilex in a_germanet.lexid2lex[ilexid]])
        # print("_synid2lex: new_set =", repr(new_set), file = sys.stderr)
        ret.append(new_set)
    return ret


def _train(a_clfs, a_pos, a_neg, a_neut):
    """Private method for training binary classifiers on synset sets

    @param a_clfs - list of ternary classifiers
    @param a_pos - set of synsets and their tf/idf vectors that
      have positive polarity
    @param a_neg - set of synsets and their tf/idf vectors that
      have negative polarity
    @param a_neut - set of synsets and their tf/idf vectors that
      have neutral polarity

    @return \c void

    """
    instances = [_flatten(inst[-1]) for inst in chain(a_pos, a_neg, a_neut)]
    trg_classes = [POSITIVE] * len(a_pos) + [NEGATIVE] * len(a_neg) \
        + [NEUTRAL] * len(a_neut)
    for iclf in a_clfs:
        print("Fitting {:s}".format(repr(iclf)), file=sys.stderr)
        iclf.fit(instances, trg_classes)


def _expand_synsets(a_germanet, a_synid2tfidf, a_seeds,
                    a_new_same, a_new_opposite, a_ext_syn_rels):
    """Extend sets of polar terms by applying custom decision function

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_seeds - set of candidate synsets
    @param a_new_same - new potential items of the same class
    @param a_new_opposite - new potential items of the opposite class
    @param a_ext_syn_rels - use extended set of synonymous relations

    @return \c void

    """
    trg_set = None
    # iterate over each synset in the seed set
    for isrc_id, _ in a_seeds:
        # obtain new synsets by following the links
        for itrg_id, irelname in a_germanet.con_relations.get(isrc_id,
                                                              [(None, None)]):
            if a_ext_syn_rels and irelname in SYNRELS:
                trg_set = a_new_same
            elif irelname in ANTIRELS:
                trg_set = a_new_opposite
            else:
                continue
            if itrg_id in a_synid2tfidf:
                trg_set.add((itrg_id, a_synid2tfidf[itrg_id]))
        # iterate over each lexeme pertaining to the source synset
        for ilex_src_id in a_germanet.synid2lexids[isrc_id]:
            # iterate over all target lexemes which the given source lexeme is
            # connected to
            for ilex_trg_id, irelname in \
                    a_germanet.lex_relations.get(ilex_src_id, [(None, None)]):
                if a_ext_syn_rels and irelname in SYNRELS:
                    trg_set = a_new_same
                elif irelname in ANTIRELS:
                    trg_set = a_new_opposite
                else:
                    continue
                # iterate over all synsets which the given target lexeme
                # pertains to
                for itrg_id in a_germanet.lexid2synids[ilex_trg_id]:
                    if itrg_id in a_synid2tfidf:
                        trg_set.add((itrg_id, a_synid2tfidf[itrg_id]))
            if not a_ext_syn_rels:
                for isynid in a_germanet.lexid2synids[ilex_src_id]:
                    if isynid in a_synid2tfidf:
                        a_new_samed.add((isynid, a_synid2tfidf[isynid]))


def _expand_seeds(a_germanet, a_synid2tfidf, a_pos, a_neg, a_neut,
                  a_ext_syn_rels):
    """Extend seed sets of polar terms by following syn/ant links.

    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_clf - classifier which makes predictions about the polarity
    @param a_pos - set of synsets and their tf/idf vectors that have positive
      polarity
    @param a_neg - set of synsets and their tf/idf vectors that have negative
      polarity
    @param a_neut - set of synsets and their tf/idf vectors that have neutral
      polarity
    @param a_ext_syn_rels - use extended set of synonymous relations

    @return \c True if sets were changed, \c False otherwise

    """
    print("Expanding seed set...", end="", file=sys.stderr)
    pos_candidates = set()
    neg_candidates = set()
    # obtain new synsets
    _expand_synsets(a_germanet, a_synid2tfidf,
                    a_pos, pos_candidates, neg_candidates, a_ext_syn_rels)
    _expand_synsets(a_germanet, a_synid2tfidf,
                    a_neg, neg_candidates, pos_candidates, a_ext_syn_rels)
    # remove from potential candidates items that are already in seed sets
    seeds = a_pos | a_neg
    pos_candidates -= seeds
    neg_candidates -= seeds
    neg_candidates -= pos_candidates
    seeds.clear()
    if not pos_candidates and not neg_candidates:
        print(" done (not changed)", file=sys.stderr)
        return False
    # merge positive and negative sets
    a_pos |= pos_candidates
    a_neg |= neg_candidates
    print(" done (pos = {:d}, neg = {:d})".format(len(a_pos), len(a_neg)),
          file=sys.stderr)
    return True


def _add_terms(a_trg, a_germanet, a_synid, a_wght):
    """Add new polar terms to the list.

    Args:
      a_trg (defaultdict): target dict for adding new terms
      a_germanet (Germanet): Germanet instance
      a_synid (str): id of synset to be added
      a_wght (float): polarity score weight of the synset to be added

    Returns:
      void:

    Notes:
      updates ``a_trg`` in place

    """
    # obtain lexemes
    for ilexid in a_germanet.synid2lexids[a_synid]:
        for ilex in a_germanet.lexid2lex[ilexid]:
            a_trg[ilex] = max(a_trg[ilex], a_wght)


def _expand_pol_lists(a_clfs, a_germanet, a_synid2tfidf,
                      a_pos, a_neg, a_neut, a_seeds):
    """Extend sets of polar terms by applying an ensemble of classifiers

    @param a_clfs - list of classifiers which makes predictions about the
      polarity
    @param a_germanet - GermaNet instance
    @param a_synid2tfidf - dictionary mapping synset id's to tf/idf vectors
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seeds - set of already included synset ids

    @return sorted list of polar terms

    """
    assert a_clfs and a_clfs[0], \
        "'a_clfs' must be a non-empty list of classifiers."
    idx2cls = a_clfs[0][0].classes_
    # check that all classifiers have the same classes
    for clf1, clf2 in a_clfs:
        assert (clf1.classes_ == idx2cls).all(), \
            "Classifier {:s} has different class2index mapping.".format(
                repr(clf1))
        assert (clf2.classes_ == idx2cls).all(), \
            "Classifier {:s} has different class2index mapping.".format(
                repr(clf2))
    icls = ""
    icls_idx = 0
    icls_score = 0.
    probs = np.zeros((1, len(idx2cls)))
    new_pos = defaultdict(lambda: 0.)
    new_pos.update({iterm: len(a_clfs) * 2 for iterm in a_pos})
    new_neg = defaultdict(lambda: 0.)
    new_neg.update({iterm: len(a_clfs) * 2 for iterm in a_neg})
    # iterate over all synsets that have tf/idf vectors
    for syn_id, tfidf_vec in a_synid2tfidf.iteritems():
        if syn_id in a_seeds:
            continue
        probs *= 0.
        for clf1, clf2 in a_clfs:
            probs += softmax(clf1.decision_function(tfidf_vec))
            probs += clf2.predict_proba(tfidf_vec)
        # make decision on the polarity class
        icls_idx = np.argmax(probs)
        icls_score = probs[0, icls_idx]
        icls = idx2cls[icls_idx]
        if icls == POSITIVE:
            _add_terms(new_pos, a_germanet, syn_id, icls_score)
        elif icls == NEGATIVE:
            _add_terms(new_neg, a_germanet, syn_id, icls_score)
    new_terms = [(iterm, itag, iwght)
                 for iterm2wght, itag in zip((new_pos, new_neg),
                                             ("positive", "negative"))
                 for iterm, iwght in iterm2wght.iteritems()]
    new_terms.sort(key=lambda el: el[-1], reverse=True)
    return new_terms


def esuli_sebastiani(a_germanet, a_pos, a_neg, a_neut, a_seed_pos,
                     a_ext_syn_rels):
    """Extend sentiment lexicons using the  method of Esuli and Sebastiani.

    @param a_germanet - GermaNet instance
    @param a_pos - set of lexemes with positive polarity
    @param a_neg - set of lexemes with negative polarity
    @param a_neut - set of lexemes with neutral polarity
    @param a_seed_pos - part-of-speech class of seed synsets ("none" for no
      restriction)
    @param a_ext_syn_rels - use extended set of synonymous relations

    @return \c void

    """
    # obtain Tf/Idf vector for each synset description
    synid2tfidf = _get_tfidf_vec(a_germanet)
    if a_seed_pos == "none":
        a_seed_pos = None
    # convert obtained lexemes to synsets
    ipos = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_pos, a_seed_pos)
    print("# of positive synsets =", len(ipos), file=sys.stderr)
    ineg = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_neg, a_seed_pos)
    ineg -= ipos
    print("# of negative synsets =", len(ineg), file=sys.stderr)
    ineut = _lexemes2synset_tfidf(a_germanet, synid2tfidf, a_neut, a_seed_pos)
    ineut -= ipos
    ineut -= ineg
    print("# of neutral synsets =", len(ineut), file=sys.stderr)
    # train two classifiers (SVC and Rocchio) on each expansion step
    clfs = [(LinearSVC(class_weight="balanced"), Rocchio()) for _ in ITERS]

    # expand seed sets and train classifiers
    prev_i = 0
    changed = True
    for j, (i, iclfs) in enumerate(zip(ITERS, clfs)):
        for _ in xrange(i - prev_i):
            changed = _expand_seeds(a_germanet, synid2tfidf,
                                    ipos, ineg, ineut, a_ext_syn_rels)
            if not changed:
                break
            ineut -= ipos
            ineut -= ineg
        # exit if the set could not be expanded
        if not changed:
            break
        prev_i = i
        # fit classifiers on the expanded seed sets
        _train(iclfs, ipos, ineg, ineut)
    # check whether classifier was trained
    j += 1
    assert j > 1, "No classifier was trained."
    clfs = clfs[:j]
    # expand resulting lists
    seeds = set(synid_vec[0] for synid_vec in chain(ipos, ineg))
    return _expand_pol_lists(clfs, a_germanet, synid2tfidf,
                             a_pos, a_neg, a_neut, seeds)
