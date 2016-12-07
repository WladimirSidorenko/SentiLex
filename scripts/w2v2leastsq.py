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

def main(argv):
    """Main method for joining word2vec and task specicifc embeddings.

    Args:
      argv (list[str]): command line arguments

    Returns:
      0 on success, non-0 otherwise

    """
    argparser = argparse.ArgumentParser(description=
                                        "Script for joining word2vec and task-"
                                        "specicifc embeddings via linear least"
                                        " squares.")
    argparser.add_argument("w2v_file",
                           help="file containing word2vec embeddings in text"
                           " format")
    argparser.add_argument("ts_file",
                           help="file containing task-specific embeddings in"
                           "plain-text format")
    args = argparser.parse_args(argv)
    w2r_w2v, M_w2v = _read_emb(args.w2v_file)
    return 0


##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
