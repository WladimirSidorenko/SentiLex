#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Script for joining word2vec and task specicifc embeddings via least squares.

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

import argparse
import codecs
import numpy as np
import sys


##################################################################
# Constants
ENCODING = "utf-8"


##################################################################
# Methods
def _read_emb(a_fname):
    """Read embedding file.

    Args:
      a_fname (str): name of the input file

    Returns:
      (np.array): embedding matrix

    """
    w2i = {}
    with codecs.open(a_fname, 'r', ENCODING) as ifile:
        iline = ifile.readline()
        rows, cols = iline.strip().split()
        M = np.empty((int(rows), int(cols)))
        for i, iline in enumerate(ifile):
            iline = iline.strip()
            if not iline:
                continue
            fields = iline.split()
            w2i[fields[0]] = i
            for j, v in enumerate(fields[1:]):
                M[i, j] = float(v)
    return w2i, M


def _compute_lstsq(w2r_w2v, M_w2v, w2r_ts, M_ts):
    """Compute least-square mapping from word2vec to task-specific embeddings.

    Args:
      w2r_w2v (dict):
      M_w2v (np.array):
      w2r_ts (dict):
      M_ts (np.array):

    Returns:
      (np.array):

    """
    # determine common keys
    cmn_trms = set(w2r_w2v.iterkeys()) & set(w2r_ts.iterkeys())
    # construct matrices with common keys
    m = len(cmn_trms)
    w2v = np.empty((m, M_w2v.shape[1]))
    ts = np.empty((m, M_ts.shape[1]))
    for i, ct in enumerate(cmn_trms):
        # copy word2vec row
        w2v[i] = M_w2v[w2r_w2v[ct]]
        # copy ts row
        ts[i] = M_ts[w2r_ts[ct]]
    X, res, _, _ = np.linalg.lstsq(w2v, ts)
    print("residuals =", res, file=sys.stderr)
    return X


def main(argv):
    """Main method for joining word2vec and task specicifc embeddings.

    Args:
      argv (list[str]): command line arguments

    Returns:
      0 on success, non-0 otherwise

    """
    argparser = argparse.ArgumentParser(description="Script for joining"
                                        " word2vec and task-specicifc"
                                        " embeddings via linear least"
                                        " squares.")
    argparser.add_argument("w2v_file",
                           help="file containing word2vec embeddings in text"
                           " format")
    argparser.add_argument("ts_file",
                           help="file containing task-specific embeddings in"
                           "plain-text format")
    args = argparser.parse_args(argv)
    w2r_w2v, M_w2v = _read_emb(args.w2v_file)
    w2r_ts, M_ts = _read_emb(args.ts_file)
    M_lst_sq = _compute_lstsq(w2r_w2v, M_w2v, w2r_ts, M_ts)

    n = len(set(w2r_w2v.iterkeys()) | set(w2r_ts.iterkeys()))
    print("{:d}\t{:d}".format(n, M_ts.shape[1]))
    vec = None
    for w, r in w2r_ts.iteritems():
        vec = M_ts[r]
        print("{:s}\t{:s}".format(w, ' '.join(
            str(v) for v in vec)).encode(ENCODING))
    for w, r in w2r_w2v.iteritems():
        if w in w2r_ts:
            continue
        vec = M_w2v[r]
        vec = M_lst_sq.dot(vec)
        print("{:s}\t{:s}".format(w, ' '.join(
            str(v) for v in vec)).encode(ENCODING))
    return 0


##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
