#!/usr/bin/env python

##################################################################
# Imports
import numpy

##################################################################
# Variables and Constants
POS = {"gut": None, "gute": None}
NEG = {"schlecht": None}

##################################################################
# Methods
def _compute_eucl_distance(a_vec1, a_vec2):
    """Compute Euclidean distance between two vectors

    @param a_vec1 - first vector
    @param a_vec2 - second vector

    @return squared Euclidean distance between two vectors
    """
    return sum((a_vec1 - a_vec2)**2)

def compute_distance(a_vecs1, a_vecs2):
    """Compute Euclidean distance between all pairs of vectors

    @param a_vecs1 - set of positive vectors
    @param a_vecs2 - set of negative vectors

    @return squared Euclidean distance between all pairs of vectors
    """
    return sum([_compute_eucl_distance(ivec1, ivec2) for ivec1 in a_vecs1 \
                    for ivec2 in a_vecs2])

def find_optimal_prj(a_dim):
    """Find projection line which optimizes difference between two vector sets

    @param a_dim - dimension of vectors

    @return 2-tuple with difference and projection line
    """
    pass

def parse_vecfile(a_fname):
    """Parse files containing word vectors

    @param a_fname - name of the wordvec file

    @return \c dimension of the vectors
    """
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        fnr = True
        toks = None
        for iline in ifile:
            iline = iline.strip()
            if fnr:
                ndim = iline.split()[-1]
                fnr = False
                continue
            elif not iline:
                continue
            toks = iline.split()
            assert len(toks) + 1 == ndim, "Wrong vector dimension"
            if toks[0] in POS:
                POS[toks[0]] = np.array([float(i) for i in toks[1:]])
            elif toks[0] in NEG:
                NEG[toks[0]] = np.array([float(i) for i in toks[1:]])
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
    ret = find_optimal_prj(ndim)

##################################################################
# Main
if __name__ == "__main__":
    main()
