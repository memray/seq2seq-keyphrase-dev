#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import nltk
import numpy
import numpy as np
import re

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
SENTENCEDELIMITER = '<eos>'
DIGIT = '<digit>'

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def prepare_text(record, tokenize_sentence = True):
    '''
    concatenate title and abstract, do sentence tokenization if needed
        As I keep most of punctuations (including period), actually I should have stopped doing sentence boundary detection
    '''
    if (tokenize_sentence):
        # replace e.g. to avoid noise for sentence boundary detection
        text = record['abstract'].replace('e.g.', 'eg')
        # pad space before and after certain punctuations [_,.<>()'%]
        title = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['title'])
        sents = [re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', s) for s in sent_detector.tokenize(text)]
        text = title + ' ' + SENTENCEDELIMITER + ' ' + (' ' + SENTENCEDELIMITER + ' ').join(sents)
    else:
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['title'] + ' . ' + record['abstract'])
    return text

def get_tokens(text, type=1):
    '''
    parse the feed-in text, filtering and tokenization
    :param text:
    :param type: 0 is old way, only keep [_<>,], do sentence boundary detection, replace digits to <digit>
                 1 is new way, keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :return: a list of tokens
    '''
    if type == 0:
        text = re.sub(r'[\r\n\t]', ' ', text)

        # tokenize by non-letters
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,]', text))
        # replace the digit terms with <digit>
        tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    elif type == 1:
        text = text.lower()
        text = re.sub(r'[\r\n\t]', ' ', text)
        # tokenize by non-letters
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))
        # replace the digit terms with <digit>
        tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens

def process_keyphrase(record):
    keyphrases = record['keyword'].lower()
    # replace abbreviations
    keyphrases = re.sub(r'\(.*?\)', ' ', keyphrases)
    # pad whitespace before and after punctuations
    keyphrases = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', keyphrases)
    # tokenize with same delimiters
    keyphrases = [filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', phrase)) for phrase in
                  keyphrases.split(';')]
    # replace digit with <digit>
    keyphrases = [[w if not re.match('^\d+$', w) else DIGIT for w in phrase] for phrase in keyphrases]

    return keyphrases
