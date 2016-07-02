#!/usr/bin/env python2.7

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from germanet import Germanet

from collections import Counter, defaultdict
from itertools import chain
from matplotlib import collections as mc

import argparse
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import sys


##################################################################
# Constants
WORDNET = "wordnet"
GERMANET = "germanet"
COLORS = ("#FFA54F", "#36648B", "#00EE76", "yellow")
POS2LABEL = {"nomen": "noun", "verben": "verb", "adj": "adjective"}
REL2COLOR = {
    "causes": "#00F5FF",
    "entails": "#00F5FF",
    "has_antonym": "#8B1A1A",
    "has_component_meronym": "#00008B",
    "has_member_meronym": "#00008B",
    "has_portion_meronym": "#00008B",
    "has_substance_meronym": "#00008B",
    "has_participle": "#FFA54f",
    "has_pertainym": "#FFFF00",
    "has_hypernym": "#8b4789",
    "has_hyponym": "#8b4789",
    "is_related_to": "#006400"
}
REL2LABEL = {
    "has_antonym": "antonym",
    "has_component_meronym": "meronym",
    "has_member_meronym": "meronym",
    "has_portion_meronym": "meronym",
    "has_substance_meronym": "meronym",
    "has_participle": "participle",
    "has_pertainym": "pertainym",
    "has_hypernym": "hypernym",
    "is_related_to": "related_to"
}
AX = plt.axes()
DE_REL_RELS = ["has_hypernym", "has_antonym", "has_pertainym", "is_related_to"]


##################################################################
# Methods
def main(a_argv):
    """Main method for visualizing WordNet databases.

    @param a_argv - command-line arguments

    @return \c 0 on success, non-\c 0 otherwise

    """
    argparser = argparse.ArgumentParser(
        description="Script for visualizing WordNet-like databases.")
    argparser.add_argument("wntype",
                           help="type of lexical database to visualize",
                           choices=(WORDNET, GERMANET))
    argparser.add_argument("path", help="path to the lexical database")
    args = argparser.parse_args(a_argv)

    # nodes' X position, Y position, and color
    _X, _Y = [], []
    POS2X = defaultdict(list)
    POS2Y = defaultdict(list)
    # pos color mapping
    POS2COLOR = {}
    # pos color mapping
    POS2CNT = Counter()
    # mapping from pos to X range
    POS2XRANGE = {}
    # mapping from pos to Y range
    POS2YRANGE = {}
    # mapping from synset id to node's index
    SYNID2NODEID = {}
    SIGMA = 10
    # populate nodes
    if args.wntype == GERMANET:
        print("Reading GermaNet synsets... ", end="", file=sys.stderr)
        igermanet = Germanet(args.path)
        print("done", file=sys.stderr)

        rel_rels = DE_REL_RELS
        # obtain available parts of speech
        POS2CNT.update(igermanet.synid2pos.itervalues())
        poses = set(igermanet.synid2pos.itervalues())
        nposes = float(len(poses))
        rpart = 500000. / nposes
        assert nposes < len(COLORS), \
            "Too few colors defined for parts of speech."
        # populate colors and ranges for parts of speech
        j = 0
        for i, ipos in enumerate(poses):
            j = i + 0.5
            POS2COLOR[ipos] = COLORS[i]
            POS2XRANGE[ipos] = j * rpart
            if i == 2:
                j = 0.5
            POS2YRANGE[ipos] = j * rpart
        # add nodes to the graph
        x = y = 0.
        invsigma = 2.
        for i, (isynid, ipos) in enumerate(igermanet.synid2pos.iteritems()):
            SYNID2NODEID[isynid] = i
            x = np.random.normal(POS2XRANGE[ipos],
                                 POS2CNT[ipos] / invsigma)
            y = np.random.normal(POS2YRANGE[ipos],
                                 POS2CNT[ipos] / invsigma)
            _X.append(x)
            POS2X[ipos].append(x)
            _Y.append(y)
            POS2Y[ipos].append(y)
        # add edges to the graph
        lines = []
        lcolors = []
        lex_rels = None
        from_idx = to_idx = x_from = x_to = y_from = y_to = 0
        for ifrom, irels in igermanet.con_relations.iteritems():
            from_idx = SYNID2NODEID[ifrom]
            lex_rels = [(to_synid, to_rel)
                        for from_lex in igermanet.synid2lexids[ifrom]
                        for to_lex, to_rel in igermanet.lex_relations[from_lex]
                        for to_synid in igermanet.lexid2synids[to_lex]]
            x_from, y_from = _X[from_idx], _Y[from_idx]
            for (ito, irel) in chain(irels, lex_rels):
                if not irel.endswith("antonym") \
                   and not irel.endswith("hypernym") \
                   and not irel.endswith("pertainym") \
                   and not irel.endswith("participle") \
                   and not irel.endswith("entails") \
                   and not irel.endswith("related_to"):
                    continue
                to_idx = SYNID2NODEID[ito]
                x_to, y_to = _X[to_idx], _Y[to_idx]
                lines.append(((x_from, y_from), (x_to, y_to)))
                lcolors.append(REL2COLOR.get(irel, "#FFFFFF"))
        # draw edges
        lc = mc.LineCollection(lines, colors=lcolors,
                               alpha=0.15, linestyle='-'
                               )
        AX.add_collection(lc)
    # draw the graph
    for ipos, x in POS2X.iteritems():
        plt.scatter(x, POS2Y[ipos], label=POS2LABEL.get(ipos, ipos),
                    c=[POS2COLOR[ipos]] * len(x))
    # add legend for edges
    handles, labels = AX.get_legend_handles_labels()
    iline = None
    for irel in rel_rels:
        iline = mlines.Line2D([], [], color=REL2COLOR[irel], linewidth=3.)
        handles.append(iline)
        labels.append(REL2LABEL[irel])
    plt.legend(handles, labels,
               loc="upper right", scatterpoints=1)
    plt.axis("off")
    plt.savefig(args.wntype + ".png", dpi=200)
    plt.show()  # display

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
