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
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['title']) + ' '+SENTENCEDELIMITER + ' ' + re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['abstract'])
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

def build_data(data, idx2word, word2idx):
    Lmax = len(idx2word)

    # don't keep the original string, or the dataset would be over 2gb
    # instance = dict(source_str=[], target_str=[], source=[], target=[], target_c=[])
    instance = dict(source=[], target=[])
    for count, pair in enumerate(data):
        source, target = pair

        # if not multi_output:
        #     A = [word2idx[w] for w in source]
        #     B = [word2idx[w] for w in target]
        #     # C = np.asarray([[w == l for w in source] for l in target], dtype='float32')
        #     C = [0 if w not in source else source.index(w) + Lmax for w in target]
        # else:
        A = [word2idx[w] if w in word2idx else word2idx['<unk>'] for w in source]
        B = [[word2idx[w] if w in word2idx else word2idx['<unk>'] for w in p] for p in target]
        # C = np.asarray([[w == l for w in source] for l in target], dtype='float32')
        C = [[0 if w not in source else source.index(w) + Lmax for w in p] for p in target]

        # actually only source,target,target_c are used in model
        # instance['source_str'] += [source]
        # instance['target_str'] += [target]
        instance['source'] += [A]
        instance['target'] += [B]
        # instance['target_c'] += [C]
        # instance['cc_matrix'] += [C]
        if count % 1000 == 0:
            print '-------------------- %d ---------------------------' % count
            print source
            print target
            print A
            print B
            print C
    return instance

def load_pairs(records, filter=False):
    wordfreq = dict()
    filtered_records = []
    pairs = []

    for id, record in enumerate(records):
        text        = prepare_text(record, tokenize_sentence = False)
        tokens      = get_tokens(text)
        keyphrases  = process_keyphrase(record)

        for w in tokens:
            if w not in wordfreq:
                wordfreq[w]  = 1
            else:
                wordfreq[w] += 1

        for keyphrase in keyphrases:
            for w in keyphrase:
                if w not in wordfreq:
                    wordfreq[w]  = 1
                else:
                    wordfreq[w] += 1

        if id % 10000 == 0:
            print('%d \n\t%s \n\t%s \n\t%s' % (id, text, tokens, keyphrases))

        if sum([len(k) for k in keyphrases]) != 0:
            ratio = float(len(record['keyword'])) / float(sum([len(k) for k in keyphrases]))
        else:
            ratio = 0
        if ( filter and ratio < 3.5 ): # usually < 4.5 is noice
            print('!' * 100)
            print('Error found')
            print('%d - title=%s, \n\ttext=%s, \n\tkeyphrase=%s \n\tkeyphrase after process=%s \n\tlen(keyphrase)=%d, #(tokens in keyphrase)=%d \n\tratio=%.3f' % (
            id, record['title'], record['abstract'], record['keyword'], keyphrases, len(record['keyword']), sum([len(k) for k in keyphrases]), ratio))
            continue

        pairs.append((tokens, keyphrases))
        filtered_records.append(record)

    return filtered_records, pairs, wordfreq
