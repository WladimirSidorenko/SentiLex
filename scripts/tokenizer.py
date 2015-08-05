#!/usr/bin/env python
# -*- coding: utf-8; -*-

"""
This code implements a basic, Twitter-aware tokenizer.

A tokenizer is a function that splits a string of text into words. In
Python terms, we map string and unicode objects into lists of unicode
objects.

There is not a single right way to do tokenizing. The best method
depends on the application.  This tokenizer is designed to be flexible
and this easy to adapt to new domains and tasks.  The basic logic is
this:

1. The tuple regex_strings defines a list of regular expression
   strings.

2. The regex_strings strings are put, in order, into a compiled
   regular expression object called word_re.

3. The tokenization is done by word_re.findall(s), where s is the
   user-supplied string, inside the tokenize() method of the class
   Tokenizer.

4. When instantiating Tokenizer objects, there is a single option:
   preserve_case.  By default, it is set to True. If it is set to
   False, then the tokenizer will downcase everything except for
   emoticons.

The __main__ method illustrates by tokenizing a few examples.

I've also included a Tokenizer method tokenize_random_tweet(). If the
twitter library is installed (http://code.google.com/p/python-twitter/)
and Twitter is cooperating, then it should tokenize a random
English-language tweet.
"""

from __future__ import unicode_literals

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2011, Christopher Potts"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "1.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"

######################################################################
import re
import os
import sys
import string
import htmlentitydefs

######################################################################
# Constants
CHAR2CHAR = {" ": " ", "&Auml;": "Ä", "&Ouml;": "Ö", "&Uuml;": "Ü", "&auml;": "ä", \
                 "&ouml;": "ö", "Ã¶": "ö", "&uuml;": "ü", "\u00F6": "ö", "\u00FC": "ü", \
                 "«": '"', "´": "'", "»": '"', "ß": "ss", "&#223;": "ss", "–": "-", \
                 "—": "-", "“": '"', "”": '"', "„": '"', "…": "...", "€": "Euro", \
                 "▸": "-", "。": "."}
CHAR2CHAR_RE = re.compile('(' + '|'.join([re.escape(ichar) for ichar in CHAR2CHAR]) + ')')

######################################################################
# The following strings are components in the regular expression
# that is used for tokenizing. It's important that phone_number
# appears first in the final regex (since it can contain whitespace).
# It also could matter that tags comes after emoticons, due to the
# possibility of having text like
#
#     <:| and some text >:)
#
# Most imporatantly, the final element should always be last, since it
# does a last ditch whitespace-based tokenization of whatever is left.

# This particular element is used in a couple ways, so we define it
# with a name:
abbreviations = ["o.k."]

emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@*\|\\]+ # mouth
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      [*]+[-_]*[*]+
      |
      [-_]+[,.]+[-_]+
      |
      (?:&?lt;|<)3
    )"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?
      \d{3}          # exchange
      [\-\s.]*
      \d{4}          # base
    )"""
    ,
    # Emoticons:
    emoticon_string
    ,
    # HTML tags:
    # r"""<[^>]+>"""
    # ,
     r"""&?(?:[lg]t|amp);"""
    ,
    # Hyperlinks:
     r"""(?:http://?|www)[\w.:?/]+\w"""
    ,
    r"""\b(?:(?:[\w]{3,5}://?|(?:www|bit)[.]|(?:\w[-\w]+[.])+(?:a(?:ero|sia|[c-gil-oq-uwxz])|b(?:iz|[abd-jmnorstvwyz])|c(?:at|o(?:m|op)|[acdf-ik-orsuvxyz])|d[dejkmoz]|e(?:du|[ceghtu])|f[ijkmor]|g(?:ov|[abd-ilmnp-uwy])|h[kmnrtu]|i(?:n(?:fo|t)|[del-oq-t])|j(?:obs|[emop])|k[eghimnprwyz]|l[abcikr-vy]|m(?:il|obi|useum|[acdeghk-z])|n(?:ame|et|[acefgilopruz])|o(?:m|rg)|p(?:ro|[ae-hk-nrstwy])|qa|r[eosuw]|s[a-eg-or-vxyz]|t(?:(?:rav)?el|[cdfghj-pr])|xxx)\b)(?:[^\s,.:;]|\.\w)*)"""
    ,
    # Twitter hashtags and special tokens:
    r"""(?:[#%]+[äöüÄÖÜß\w_][äöüÄÖÜß\w'_-]*[äöüÄÖÜß\w_])"""
    ,
    # Twitter username:
    r"""(?:@+[äöüÄÖÜß\w_]+)"""
    ,
    # Final full stop
     r"""(?:[.!?]+$)"""
    ,
    # Abbreviations
     r"""(?:{abbrev})""".format(abbrev = '|'.join([re.escape(abbr) for abbr in \
                                                       set(abbreviations)]))
    ,
     r"""(?:\b[A-zÖöÄäÜü]+\.)(?!$|\s*[."])"""
    ,
    # Remaining word types:
    r"""
    (?:[+\-]?\d\d?\d?.\s)  # ordinal number
    |
    (?:[+\-]?\d+(?:[,/.:-]\d+)?[+\-]?)  # Numbers, including ordinals, fractions, decimals.
    |
    (?:\.(?:\s*\.){{1,}})          # Ellipsis dots.
    |
    (?:{punct})                  # punctuation
    |
    (?:\w'[\w]+)                   # French apostrophe
    |
    (?:[^{punct}{blanc}]+)         # Everything else that isn't whitespace or punctuation mark.
    """.format(punct = '|'.join([re.escape(c) for c in string.punctuation]), \
                                    blanc = string.whitespace)
    )

######################################################################
# This is the core tokenizing regex:
word_re = re.compile(r"""(%s)""" % '|'.join(regex_strings), re.VERBOSE | re.I | re.U)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.U )

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"

######################################################################
# Class
class Tokenizer:
    def __init__(self, preserve_case = True, return_offsets = False):
        """
        Class constructor

        @param preserve_case - keep case of input string unchanged
        @param return_offsets - return character offsets of split words
        """
        ## keep case of input string unchanged
        self.preserve_case  = preserve_case
        ## return character offsets of split words
        self.return_offsets = return_offsets

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object

        Value: a tokenize list of strings; concatenating this list returns the
        original string if preserve_case=False
        """
        s = CHAR2CHAR_RE.sub(lambda mobj: CHAR2CHAR.get(mobj.group(1), ""), s.strip())
        # Tokenize:
        s = self.__html2unicode(s)
        words = word_re.findall(s)

        if self.return_offsets:
            offsets = self.__get_offsets__(s, words)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:
            words = map((lambda x : x if emoticon_re.search(x) else x.lower()), words)
        if self.return_offsets:
            return zip(words, offsets)
        else:
            return words

    def __html2unicode(self, s):
        """
        Internal metod that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, unichr(entnum))
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = filter((lambda x : x != amp), ents)
        for ent in ents:
            entname = ent[1:-1].lower()
            if entname == "lt" or entname == "gt":
                continue
            try:
                s = s.replace(ent, unichr(htmlentitydefs.name2codepoint[entname]))
            except:
                pass
            s = s.replace(amp, " and ")
        return s

    def __get_offsets__(self, s, words):
        """
        Calculate positions at which each word in words starts in string s.

        It is not an optimal solution but a quick remedy.
        """
        offsets = []

        s_offset = 0
        slen  = len(s)
        wlen  = 0
        for w in words:
            wlen = len(w)
            mobj = s
            while s and s[:wlen] != w:
                s = s[1:]
                s_offset += 1
            if s_offset < slen:
                offsets.append((s_offset, wlen))
                s = s[wlen:]
                s_offset += wlen
        assert len(offsets) == len(words)
        return offsets
