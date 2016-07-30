#!/usr/bin/env python2.7

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from germanet import Germanet
from wordnet import Wordnet

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
REL2COLOR = {
    # GermaNet
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
    "is_related_to": "#006400",
    # WordNet
    "Hyponym": "#8b4789",
    "Instance Hyponym": "#8b4789",
    "Antonym": "#8B1A1A",
    "Member holonym": "#00008B",
    "Part holonym": "#00008B",
    "Substance holonym": "#00008B",
    "Verb Group": "#00CD00",
    "Member meronym": "#00008B",
    "Part meronym": "#00008B",
    "Substance meronym": "#00008B",
    "Similar to": "#FF7256",
    "Entailment": "#00F5FF",
    "Derivationally related form": "#006400",
    "Member of this domain - TOPIC": "#EE82EE",
    "Member of this domain - REGION": "#EE82EE",
    "Member of this domain - USAGE": "#EE82EE",
    "Domain of synset - TOPIC": "#EE82EE",
    "Domain of synset - REGION": "#EE82EE",
    "Domain of synset - USAGE": "#EE82EE",
    "Participle of verb": "#FFA54F",
    "Attribute": "#FFA500",
    "Cause": "#00F5FF",
    "Hypernym": "#8b4789",
    "Instance Hypernym": "#8b4789",
    "Derived from adjective": "#FFFF00",
    "Also see": "#006400"
}
REL2LABEL = {
    # GermaNet
    "has_antonym": "antonym",
    "has_component_meronym": "meronym",
    "has_member_meronym": "meronym",
    "has_portion_meronym": "meronym",
    "has_substance_meronym": "meronym",
    "has_participle": "participle",
    "has_pertainym": "pertainym",
    "has_hypernym": "hypernym",
    "has_hyponym": "hyponym",
    "is_related_to": "related_to",
    # WordNet
    "Hyponym": "hyponym",
    "Instance Hyponym": "hyponym",
    "Antonym": "antonym",
    "Member holonym": "holonym",
    "Part holonym": "holonym",
    "Substance holonym": "holonym",
    "Verb Group": "verb group",
    "Member meronym": "meronym",
    "Part meronym": "meronym",
    "Substance meronym": "meronym",
    "Similar to": "similar to",
    "Entailment": "entailment",
    "Derivationally related form": "related_to",
    "Member of this domain - TOPIC": "domain member",
    "Member of this domain - REGION": "domain member",
    "Member of this domain - USAGE": "domain member",
    "Domain of synset - TOPIC": "domain",
    "Domain of synset - REGION": "domain",
    "Domain of synset - USAGE": "domain",
    "Participle of verb": "participle",
    "Attribute": "attribute",
    "Cause": "cause",
    "Hypernym": "hypernym",
    "Instance Hypernym": "hypernym",
    "Derived from adjective": "derived_from",
    "Also see": "also see"
}

AX = plt.axes()
POS2COLOR = {"verben": "#00EE76", "v": "#00EE76",
             "nomen": "#36648B", "n": "#36648B",
             "adj": "#FFA54F", "a": "#FFA54F",
             "r": "#97FFFF", "s": "#FF4500"}
POS2LABEL = {"nomen": "noun", "n": "noun",
             "verben": "verb", "v": "verb",
             "adj": "adjective", "a": "adjective",
             "r": "adverb", "s": "adjective satellite"}
_POS2X = {"adj": 0, "a": 0,
          "nomen": 1, "n": 1,
          "verben": 2, "v": 2,
          "r": 0, "s": 1}
_POS2Y = {"adj": 0, "a": 0,
          "nomen": 1, "n": 1.5,
          "verben": 0, "v": 0,
          "r": 2.5, "s": 0.35}
DE_REL_RELS = ["has_hyponym", "has_antonym",
               "has_pertainym", "is_related_to",
               "has_participle"]
EN_REL_RELS = ["Hyponym", "Instance Hyponym", "Antonym",
               "Derived from adjective", "Derivationally related form",
               "Participle of verb"]


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
    POS2CNT = Counter()
    # mapping from pos to X range
    POS2XRANGE = {}
    # mapping from pos to Y range
    POS2YRANGE = {}
    # mapping from synset id to node's index
    SYNID2NODEID = {}
    SIGMA = 10
    # line collection to be initialized later
    lc = None
    # populate nodes
    if args.wntype == GERMANET:
        print("Reading GermaNet synsets... ", end="", file=sys.stderr)
        inet = Germanet(args.path)
        print("done", file=sys.stderr)
        rel_rels = DE_REL_RELS
    else:
        print("Reading WordNet synsets... ", end="", file=sys.stderr)
        inet = Wordnet(args.path)
        print("done", file=sys.stderr)
        rel_rels = EN_REL_RELS

    # obtain available parts of speech
    POS2CNT.update(inet.synid2pos.itervalues())
    poses = set(inet.synid2pos.itervalues())
    nposes = float(len(poses))
    rpart = 500000. / min(3, nposes)
    # populate colors and ranges for parts of speech
    x = y = 0
    for ipos in poses:
        x = _POS2X[ipos]
        y = _POS2Y[ipos]
        POS2XRANGE[ipos] = x * rpart
        POS2YRANGE[ipos] = y * rpart
    # add nodes to the graph
    x = y = 0.
    invsigma = 2.
    if args.wntype == WORDNET:
        assert ("00704270", "s") in inet.synid2pos, \
            "('00704270', 's') is missing"
    for i, (isynid, ipos) in enumerate(inet.synid2pos.iteritems()):
        # print("isynid =", repr(isynid), file=sys.stderr)
        # sys.exit(66)
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
    if args.wntype == GERMANET:
        iterrels = inet.con_relations.iteritems()
    else:
        iterrels = inet.relations.iteritems()
    for ifrom, irels in iterrels:
        # print("ifrom =", repr(ifrom), file=sys.stderr)
        # sys.exit(66)
        from_idx = SYNID2NODEID[ifrom]
        if args.wntype == GERMANET:
            lex_rels = [(to_synid, to_rel)
                        for from_lex in inet.synid2lexids[ifrom]
                        for to_lex, to_rel in inet.lex_relations[from_lex]
                        for to_synid in inet.lexid2synids[to_lex]]
        else:
            lex_rels = []
        x_from, y_from = _X[from_idx], _Y[from_idx]
        for (ito, irel) in chain(irels, lex_rels):
            # print("irel: irel = {:s} {:d}".format(repr(irel),
            #                                       irel in rel_rels),
            #       file=sys.stderr)
            if not irel in rel_rels:
                continue
            # print("rel: ifrom = {:s}, irels = {:s}".format(repr(ifrom),
            #                                                repr(irels)),
            #       file=sys.stderr)
            if ito not in SYNID2NODEID and ito[-1] == 'a':
                to_idx = SYNID2NODEID[(ito[0], 's')]
            else:
                to_idx = SYNID2NODEID[ito]
            x_to, y_to = _X[to_idx], _Y[to_idx]
            lines.append(((x_from, y_from), (x_to, y_to)))
            lcolors.append(REL2COLOR.get(irel, "#FFFFFF"))
    # draw edges
    lc = mc.LineCollection(lines, colors=lcolors,
                           alpha=0.15, linestyle='-'
                           )
    # draw the graph
    AX.add_collection(lc)
    for ipos, x in POS2X.iteritems():
        plt.scatter(x, POS2Y[ipos], label=POS2LABEL.get(ipos, ipos),
                    c=[POS2COLOR[ipos]] * len(x))
    # add legend for edges
    handles, labels = AX.get_legend_handles_labels()
    iline = ilabel = None
    known_labels = set()
    for irel in rel_rels:
        iline = mlines.Line2D([], [], color=REL2COLOR[irel], linewidth=3.)
        ilabel = REL2LABEL[irel]
        if ilabel in known_labels:
            continue
        handles.append(iline)
        labels.append(ilabel)
        known_labels.add(ilabel)
    plt.legend(handles, labels,
               loc="upper right", scatterpoints=1)
    plt.axis("off")
    plt.savefig(args.wntype + ".png", dpi=200)
    plt.show()  # display

##################################################################
# Main
if __name__ == "__main__":
    main(sys.argv[1:])
