#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""
Implementation of Trie data structure

Constants:
SPACE_RE - regular expression matching continuous runs of space characters
FINAL_SPACE_RE - regular expression matching leading and trailing white spaces
ANEW - flag indicating that the search should start anew
CONTINUE - flag indicating that the search should continue where it stopped

Classes:
State - single trie state with associated transitions
Trie - implementation of Trie data structure

@author = Uladzimir Sidarenka <sidarenk AT uni DASH potsdam DOT de>

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function
import re
import sys

##################################################################
# Variable and Constants
PUNCT_RE = re.compile("[#,.]")
SPACE_RE = re.compile("\s\s+", re.L)
FINAL_SPACE_RE = re.compile("(:?^\s+|\s+$)")
ANEW = 0
CONTINUE = 1


##################################################################
# Methods
def normalize_string(a_string, a_ignorecase=False):
    """Clean input string and return its normalized version

    @param a_string - string to clean
    @param a_ignorecase - boolean indication whether string should be
      lowercased

    @return normalized string

    """
    a_string = PUNCT_RE.sub("", a_string)
    a_string = SPACE_RE.sub(' ', a_string)
    a_string = FINAL_SPACE_RE.sub("", a_string)
    if a_ignorecase:
        a_string = a_string.lower()
    return a_string


##################################################################
# Classes
class MatchObject(object):
    """
    Object returned by Trie class on successful match

    Instance variables:
    start - start index of the match
    end - end index of the match

    Methods:
    __init__() - class constructor
    update() - update end index of match
    """

    def __init__(self, a_start):
        """
        Class constructor

        @param a_start - start index of match
        """
        ## start index of match
        self.start = a_start
        ## end index of match
        self.end = -1

    def update(self, a_end):
        """
        Class constructor

        @param a_end - new end index of match

        @return \c void
        """
        self.end = a_end


class State(object):
    """
    Single Trie state with associated transitions

    Instance variables:
    classes - custom classes associated with the current state
    final - boolean flag indicating whether state is final
    transitions - set of transitions triggered by the state

    Methods:
    __init__() - class constructor
    add_transition() - add new transition from that state
    check() - check transitions associated with the given character
    """

    def __init__(self, a_final=False, a_class=None):
        """
        Class constructor

        @param a_final - boolean flag indicating whether the state is
                         final or not
        @param a_class - custom class associated with the final state
        """
        ## custom classes associated with the current state
        self.classes = set([])
        if a_class is not None:
            self.classes.add(a_class)
        ## boolean flag indicating whether the state is final
        self.final = a_final
        ## set of transitions triggered by the given state
        self.transitions = dict()

    def add_transition(self, a_char):
        """
        Add new transition outgoing from that state

        @param a_char - character triggerring that trnasition

        @return address of the target state of that transition
        """
        if a_char not in self.transitions:
            self.transitions[a_char] = State()
        return self.transitions[a_char]

    def check(self, a_char):
        """
        Check transitions associated with the given character

        @param a_char - character whose associated transitions should
        be checked

        @return \c set of target states triggered by character
        """
        if a_char in self.transitions:
            return self.transitions[a_char]
        return None


class Trie(object):
    """
    Implementation of trie data structure

    Instance variables:
    ignorecase - boolean flag indicating whether the case
                 should be ignored
    active_state - set of currently active Trie states

    Methods:
    __init__() - class constructor
    add() - add new string to the trie
    match() - compare given string against the trie
    """

    def __init__(self, a_ignorecase=False):
        """
        Class constructor

        @param a_ignorecase - boolean flag indicating whether the case
                              should be ignored
        """
        ## boolean flag indicating whether character case should be ignored
        self.ignorecase = a_ignorecase
        self._init_state = State()
        ## set of currently active Trie states
        self.active_states = set([])

    def add(self, a_string, a_class=0):
        """
        Add new string to the trie

        @param a_string - string to be added
        @param a_class - optional custom class associated with that string

        @return \c void
        """
        # normalize string
        a_string = normalize_string(a_string, self.ignorecase)
        # successively add states
        astate = self._init_state
        for ichar in a_string:
            astate = astate.add_transition(ichar)
        astate.final = True
        astate.classes.add(a_class)

    def match(self, a_strings, a_start=-1, a_reset=ANEW):
        """
        Compare given strings against the Trie

        @param a_strings - list of strings to be matched
        @param a_start - (optional) start index of the string
        @param a_reset - (optional) flag indicating whether search
                         should start anew or continue

        @return \c True if at least one match succeeded
        """
        a_strings = [normalize_string(istring, self.ignorecase)
                     if istring != ' ' else istring
                     for istring in a_strings if istring is not None]
        if a_reset == ANEW and a_strings:
            self.active_states = set([(self._init_state, a_start, -1)])
        else:
            self.active_states.add((self._init_state, a_start, -1))
        # set of tuples with states and associated match objects
        ret = set()
        status = False
        # print("active_states =", repr(self.active_states), file = sys.stderr)
        for istring in a_strings:
            # print("istring =", repr(istring), file = sys.stderr)
            if istring is None:
                continue
            for istate, istart, iend in self.active_states:
                for ichar in istring:
                    # print("matching char:", repr(ichar), file = sys.stderr)
                    # print("istate.transitions:", repr(istate.transitions),
                    # file = sys.stderr)
                    istate = istate.check(ichar)
                    if istate is None:
                        break
                    # print("char matched", file = sys.stderr)
                    # print("istate =", repr(istate), file = sys.stderr)
                else:
                    if istate.final:
                        status = True
                    # print("adding istate to ret:", repr(ret), file =
                    # sys.stderr)
                    ret.add((istate, istart,
                             a_start if a_start >= 0 else iend))
                    # print("istate added:", repr(ret), file = sys.stderr)
            # if ret:
            #     break
            # print("ret =", repr(ret), file = sys.stderr)
        self.active_states = ret
        return status

    def __unicode__(self):
        """
        Return a unicode representation of the given trie

        @return unicode object representing the trie in graphviz format
        """
        ret = """digraph Trie {
        size="6,6";
        rankdir = "LR";
        {node [color=black,fillcolor=palegreen,style=filled,forcelabels=true];
        """
        istate = None
        scnt = tcnt = rels = ""
        state_cnt = 0
        state2cnt = {self._init_state: str(state_cnt)}
        new_states = [self._init_state]
        visited_states = set()
        while new_states:
            istate = new_states.pop()
            scnt = state2cnt[istate]
            if istate.final:
                ret += '{:s} [shape=box,fillcolor=' \
                       'lightblue,label="{:s}"];\n'.format(
                           scnt, ", ".join(istate.classes))
            else:
                ret += "{:s} [shape=circle];\n".format(scnt)
            for jchar, jstate in istate.transitions.iteritems():
                if jstate not in state2cnt:
                    state_cnt += 1
                    state2cnt[jstate] = str(state_cnt)
                if jstate not in visited_states:
                    new_states.append(jstate)
                rels += scnt + "->" + state2cnt[jstate] \
                    + """[label="'{:s}'"];\n""".format(jchar)
            visited_states.add(istate)
        ret += "}\n" + rels + "}\n"
        return ret
