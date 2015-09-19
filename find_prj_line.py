#!/usr/bin/env python

##################################################################
# Imports
from __future__ import print_function

from random import random
import codecs
import numpy as np
import sys

##################################################################
# Variables and Constants
ENCODING = "utf-8"
POS = {"gut": None, "gute": None}
NEG = {"schlecht": None}

##################################################################
# Methods
def _get_vec_len(a_vec):
    """Return length of the vector

    @param a_vec - vector whose length should be computed

    @return vector's length
    """
    return np.sqrt(sum([i**2 for i in a_vec]))

def _compute_eucl_distance(a_vec1, a_vec2):
    """Compute Euclidean distance between two vectors

    @param a_vec1 - first vector
    @param a_vec2 - second vector

    @return squared Euclidean distance between two vectors
    """
    return sum((a_vec1 - a_vec2)**2)

def compute_distance(a_vecs1, a_vecs2):
    """Compute Euclidean distance between all pairs of vectors

    Compute \sum_{p^{+} \in P^{+}}\sum_{p^{-} \in P^{-}}||p^{+} - p^{-}||^{2}

    @param a_vecs1 - set of positive vectors
    @param a_vecs2 - set of negative vectors

    @return squared Euclidean distance between all pairs of vectors
    """
    return sum([_compute_eucl_distance(ivec1, ivec2) for ivec1 in a_vecs1 \
                    for ivec2 in a_vecs2])

def _project_vec(a_vec, a_norm, a_prj_line):
    """Project original vector on projection line

    @param a_vec - vector whoch should e projected
    @param a_norm - square length of projection line
    @param a_prj_line - projection line

    @return a_vec's projection on a_prj_line
    """
    print("_project_vec: a_vec =", repr(a_vec), file = sys.stderr)
    print("_project_vec: a_prj_line =", repr(a_prj_line), file = sys.stderr)
    print("_project_vec: a_norm =", repr(a_norm), file = sys.stderr)
    print("_project_vec: np.dot() =", repr(np.dot(a_vec, a_prj_line)), file = sys.stderr)
    print("_project_vec: projection =", \
              repr((np.dot(a_vec, a_prj_line) / a_norm) * a_prj_line), file = sys.stderr)
    return (np.dot(a_vec, a_prj_line) / a_norm) * a_prj_line

def _project(a_pos_set, a_neg_set, a_prj_line):
    """Project original vector sets on the projection line

    @param a_pos_set - set of positive vectors
    @param a_neg_set - set of negative vectors
    @param a_prj_line - projection line

    @return 2-tuple with sets of projected positive and negative vectors
    """
    idiv = sum(a_prj_line ** 2)
    assert idiv != 0, "Projection vector cannot be zero vector."
    vecs1 = [_project_vec(ivec, idiv, a_prj_line) for ivec in a_pos_set]
    vecs2 = [_project_vec(ivec, idiv, a_prj_line) for ivec in a_neg_set]
    return (vecs1, vecs2)

def _compute_gradient(a_pos_vecs, a_pos_prjctd, a_neg_vecs, \
                          a_neg_prjctd, a_prj_line, a_gradient):
    """Compute gradient of distance function wrt projection line

    @param a_pos_vecs - set of positive vectors
    @param a_pos_prjctd - set of projected positive vectors
    @param a_neg_vecs - set of negative vectors
    @param a_pos_prjctd - set of projected negative vectors
    @param a_prj_line - current projection line
    @param a_gradient - gradient vector to be updated

    @return gradient vector
    """
    # zero-out the gradient vector
    update = None
    idiv = sum(a_prj_line ** 2)
    idiv_ones = idiv * np.ones(a_prj_line.size)
    idiv_squared = idiv ** 2
    prj_squared = a_prj_line ** 2
    normalized_prj = a_prj_line / idiv
    assert idiv != 0, "Projection vector cannot be zero vector."
    a_gradient = np.array([0 for _ in a_gradient])
    for pos_vec, pos_prjctd in zip(a_pos_vecs, a_pos_prjctd):
        for neg_vec, neg_prjctd in zip(a_neg_vecs, a_neg_prjctd):
            update = pos_prjctd - neg_prjctd
            update *= (pos_vec * idiv - 2 * np.dot(pos_vec, a_prj_line) * a_prj_line) / idiv_squared + \
                np.dot(pos_vec, a_prj_line) / idiv - \
                (neg_vec * idiv - 2 * np.dot(neg_vec, a_prj_line) * a_prj_line) / idiv_squared - \
                np.dot(neg_vec, a_prj_line) / idiv
            a_gradient += update
    # since we have a quadratic function, the gradient has
    # coefficient two
    return 2 * a_gradient

def find_optimal_prj(a_dim):
    """Find projection line that maximizes distance between vectors

    @param a_dim - dimension of vectors

    @return 2-tuple with projection line and cost
    """
    DELTA = 1e-10               # cost difference
    ALPHA = 1                   # learning rate
    n = 0                       # current iteration
    max_n = 1000                # maximum number of iterations
    inf = float("inf")
    ipos = ineg = None
    prev_dist = dist = inf
    prj_line = np.array([1 for _ in a_dim])
    # prj_line = np.array([random() for i in xrange(a_dim)])
    gradient = np.array([1 for i in xrange(a_dim)])
    while (prev_dist == inf or prev_dist - dist > DELTA) and n < max_n:
        prev_dist = dist
        # normalize length of projection line
        prj_line /= _get_vec_len(prj_line)
        # project word vectors on the guessed polarity line
        # print("POS = ", repr(POS), file = sys.stderr)
        # print("NEG = ", repr(NEG), file = sys.stderr)
        ipos, ineg = _project(POS.itervalues(), NEG.itervalues(), prj_line)
        # compute distance between posiive and negative vectors
        # print("ipos = ", repr(ipos), file = sys.stderr)
        # print("ineg = ", repr(ineg), file = sys.stderr)
        dist = compute_distance(ipos, ineg)
        print("prev_dist = {:f}".format(dist), file = sys.stderr)
        print("dist = {:f}".format(dist), file = sys.stderr)
        # update polarity line
        prj_line += ALPHA * _compute_gradient(POS.itervalues(), ipos, \
                                                  NEG.itervalues(), ineg, \
                                                  prj_line, gradient)
        print("prj_line = ", prj_line, file = sys.stderr)
        n += 1
    return (prj_line, dist)

def parse_vecfile(a_fname):
    """Parse files containing word vectors

    @param a_fname - name of the wordvec file

    @return \c dimension of the vectors
    """
    global POS, NEG
    ivec = None
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        fnr = True
        toks = None
        for iline in ifile:
            iline = iline.strip()
            if fnr:
                ndim = int(iline.split()[-1])
                fnr = False
                continue
            elif not iline:
                continue
            toks = iline.split()
            assert (len(toks) - 1) == ndim, "Wrong vector dimension: {:d}".format(\
                len(toks) - 1)
            if toks[0] in POS:
                ivec = np.array([float(i) for i in toks[1:]])
                ivec /= _get_vec_len(ivec)
                POS[toks[0]] = ivec
            elif toks[0] in NEG:
                ivec = np.array([float(i) for i in toks[1:]])
                ivec /= _get_vec_len(ivec)
                NEG[toks[0]] = ivec
    # prune words for which there were no vectors
    POS = {iword: ivec for iword, ivec in POS.iteritems() if ivec is not None}
    NEG = {iword: ivec for iword, ivec in NEG.iteritems() if ivec is not None}
    return ndim

def main():
    """Main method for finding optimal projection line

    @return square sum of the distances between positive and negative
    words projected on the line
    """
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("vec_file", help = "file containing vectors")
    args = argparser.parse_args()
    ndim = parse_vecfile(args.vec_file)
    prj_line, ret = find_optimal_prj(ndim)
    print("ret =", str(ret))

##################################################################
# Main
if __name__ == "__main__":
    main()
