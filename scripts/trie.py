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
from __future__ import unicode_literals
import re

##################################################################
# Variable and Constants
SPACE_RE = re.compile("\s\s+", re.L)
FINAL_SPACE_RE = re.compile("(:?^\s+|\s+$)")
ANEW = 0
CONTINUE = 1

##################################################################
# Classes
class State(object):
    """
    Single trie state with associated transitions

    Class constants:
    EMPTY_SET - set containing no elements

    Instance variables: classes - custom classes associated with a
    final - boolean flag indicating whether state is final
    transitions - set of transitions triggered by the state

    Methods:
    __init__() - class constructor
    add_transition() - add new transition from that state
    check() - check transitions associated with the given character
    """

    EMPTY_SET = frozenset()

    def __init__(self, a_final = False, a_class = None):
        """
        Class constructor

        @param a_final - boolean flag indicating whether the state is
                         final or not
        @param a_class - custom class associated with the final state
        """
        self.classes = set([])
        if a_class is not None:
            self.classes.add(a_class)
        self.final = a_final
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
        return EMPTY_SET

class Trie(object):
    """
    Implementation of trie data structure

    Instance variables:
    ignorecase - boolean flag indicating whether the case
                 should be ignored
    active_state - currently active state

    Methods:
    __init__() - class constructor
    add() - add new string to the trie
    match() - compare given string against the trie
    """

    def __init__(self, a_ignorecase = False):
        """
        Class constructor

        @param a_ignorecase - boolean flag indicating whether the case
                              should be ignored
        """
        self.ignorecase = a_ignorecase
        self._init_state = State()
        self.active_states = set([self._init_state])
        self._working_states = set()

    def add(self, a_string, a_class = 0):
        """
        Add new string to the trie

        @param a_string - string to be added
        @param a_class - optional custom class associated with that string

        @return \c void
        """
        # adjust string
        a_string = SPACE_RE.sub(' ', a_string)
        a_string = FINAL_SPACE_RE.sub("", a_string)
        if self.ignorecase:
            a_string = a_string.lower()
        # successively add states
        astate = self._init_state
        for ichar in a_string:
            astate = astate.add_transition(ichar)
        astate.final = True
        astate.classes.add(a_class)

    def match(self, a_string, a_reset = ANEW):
        """
        Comapre given string against the class

        @param a_string - string to be added
        @param a_reset - flag indicating whether search should start anew or continue

        @return \c class(es) of the input string or None if there was no match
        """
        if a_reset == ANEW:
            self.active_states = set([self._init_state])
        for ichar in a_string:
            self._working_states.clear()
            for astate in self.active_states:
                self._working_states |= astate.check(ichar)
            self._working_states, self.active_states = self.active_states, self._working_states
        ret = set()
        for astate in self.active_states:
            if astate.final:
                ret |= astate.classes
        return ret

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
                ret += '{:s} [shape=box,fillcolor=lightblue,label="{:s}"];\n'.format(scnt, ", ".join(istate.classes))
            else:
                ret += "{:s} [shape=circle];\n".format(scnt)
            for jchar, jstate in istate.transitions.iteritems():
                if jstate not in state2cnt:
                    state_cnt += 1
                    state2cnt[jstate] = str(state_cnt)
                if jstate not in visited_states:
                    new_states.append(jstate)
                rels += scnt + "->" + state2cnt[jstate] + """[label="'{:s}'"];\n""".format(jchar)
            visited_states.add(istate)
        ret += "}\n" + rels + "}\n"
        return ret
