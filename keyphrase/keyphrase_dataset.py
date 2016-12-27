# coding=utf-8
import json
import sys
import time

import nltk
import numpy as np

import config
from emolga.dataset.build_dataset import *
import re

MULTI_OUTPUT = False
TOKENIZE_SENTENCE = True

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
wordfreq = dict()
SENTENCEDELIMITER = '<eos>'
DIGIT = '<digit>'

def get_tokens(text, type=1):
    '''

    :param text:
    :param type: 0 is old way, only keep [_<>,], do sentence boundary detection, replace digits to <digit>
                 1 is new way, keep [_<>,\(\)\.\'%], replace digits to <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :return:
    '''
    if type == 0:
        text = re.sub(r'[\r\n\t]', ' ', text)

        text = text.replace('e.g.', 'eg')
        sents = [re.sub(r'[_<>,]', ' \g<0> ', s) for s in sent_detector.tokenize(text)]

        text = (' ' + SENTENCEDELIMITER + ' ').join(sents)
        text = text.lower()

        # tokenize by non-letters
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,]', text))
        # replace the digit terms with <digit>
        tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    elif type == 1:
        text = text.lower()
        text = re.sub(r'[\r\n\t]', ' ', text)
        text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
        # tokenize by non-letters
        tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))
        # replace the digit terms with <digit>
        tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens


def load_data(input_path, tokenize_sentence=True):
    global wordfreq

    # data set
    pairs = []
    f     = open(input_path, 'r')
    records = json.load(f)

    for id, record in enumerate(records):
        # record['abstract'] = record['abstract'].encode('utf8')
        # record['title'] = record['abstract'].encode('utf8')
        # record['keyword'] = record['abstract'].encode('utf8')

        # record = json.loads(record)
        if record['abstract'].find('(svm)') > -1 or record['keyword'].find('(svm)') > -1:
            print(record)

        if (tokenize_sentence):
            text = record['abstract'].replace('e.g.','eg')
            title =  re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['title'])
            sents = [re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', s) for s in sent_detector.tokenize(text)]
            text =  title + ' '+SENTENCEDELIMITER+' ' + (' '+SENTENCEDELIMITER+' ').join(sents)
        else:
            text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', record['title'] + ' . ' + record['abstract'])

        text = text.lower()
        # text = re.sub(r'[_<>,()\.\']', ' \g<0> ', text)

        # tokenize by non-letters
        text = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', text))
        # replace the digit terms with <digit>
        text = [w if not re.match('^\d+$', w) else DIGIT for w in text]

        for w in text:
            if w not in wordfreq:
                wordfreq[w]  = 1
            else:
                wordfreq[w] += 1

        # store the terms of outputs
        keyphrases = record['keyword'].lower()
        keyphrases = re.sub(r'\(.*?\)', ' ', keyphrases)
        keyphrases = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', keyphrases)
        # tokenize with same delimiters
        keyphrases = [filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,\(\)\.\'%]', phrase)) for phrase in keyphrases.split(';')]
        # replace digit with <digit>
        keyphrases = [[w if not re.match('^\d+$', w) else DIGIT for w in phrase] for phrase in keyphrases]

        for keyphrase in keyphrases:
            for w in keyphrase:
                if w not in wordfreq:
                    wordfreq[w]  = 1
                else:
                    wordfreq[w] += 1

        # if multi_output:
        #     pairs.append((text, keyphrases))
        # else:
        #     pairs.extend([(text,k) for k in keyphrases])
        pairs.append((text, keyphrases))
        if id % 10000 == 0:
            print('%d \n\t%s \n\t%s' % (id, text, keyphrases))

    return pairs


def build_dict(wordfreq):
    word2idx = dict()
    word2idx['<eol>'] = 0
    word2idx['<unk>'] = 1
    start_index = 2

    # sort the vocabulary (word, freq) from low to high
    wordfreq = sorted(wordfreq.items(), key=lambda a: a[1], reverse=True)

    # create word2idx
    for w in wordfreq:
        word2idx[w[0]] = start_index
        start_index += 1

    # create idx2word
    idx2word = {k: v for v, k in word2idx.items()}
    Lmax = len(idx2word)
    # for i in xrange(Lmax):
    #     print idx2word[i].encode('utf-8')

    return idx2word, word2idx

def build_data(data, idx2word, word2idx):
    Lmax = len(idx2word)

    instance = dict(source_str=[], target_str=[], source=[], target=[], target_c=[])
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
        instance['source_str'] += [source]
        instance['target_str'] += [target]
        instance['source'] += [A]
        instance['target'] += [B]
        instance['target_c'] += [C]
        # instance['cc_matrix'] += [C]
        if count % 10000 == 0:
            print '-------------------- %d ---------------------------' % count
            print source
            print target
            print A
            print B
            print C
    return instance

def load_data_and_dict(training_dataset, testing_dataset):
    global wordfreq
    # do not do one2one as it consumes too much memory, parse to one2one before training
    train_pairs = load_data(training_dataset)
    test_pairs = load_data(testing_dataset)
    print('read dataset done.')

    idx2word, word2idx = build_dict(wordfreq)
    print('build dicts done.')

    # use character-based model [on]
    # use word-based model     [off]
    train_set = build_data(train_pairs, idx2word, word2idx)
    test_set = build_data(test_pairs, idx2word, word2idx)


    print('Train pairs: %d' % len(train_pairs))
    print('Test pairs:  %d' % len(test_pairs))
    print('Dict size:   %d' % len(idx2word))
    return train_set, test_set, idx2word, word2idx


def load_additional_testing_data(testing_path, idx2word, word2idx):
    test_pairs = load_data(testing_path)
    test_set = build_data(test_pairs, idx2word, word2idx)
    return test_set

if __name__ == '__main__':
    # config = config.setup_keyphrase_all()
    config = config.setup_keyphrase_all()

    start_time = time.clock()
    train_set, test_set, idx2word, word2idx = load_data_and_dict(config['training_dataset'], config['testing_dataset'])
    serialize_to_file([train_set, test_set, idx2word, word2idx], config['dataset'])
    print('Finish processing and dumping: %d seconds' % (time.clock()-start_time))

    # export vocabulary to file for manual check
    wordfreq = sorted(wordfreq.items(), key=lambda a: a[1], reverse=True)
    serialize_to_file(wordfreq, config['voc'])
    with open(config['path']+'/dataset/keyphrase/voc_list.json', 'w') as voc_file:
        str = ''
        for w,c in wordfreq:
            str += '%s\t%d\n' % (w,c)
        voc_file.write(str)


    # train_set, test_set, idx2word, word2idx = deserialize_from_file(config['dataset'])
    # print('Load successful: vocsize=%d'% len(idx2word))

    # count_dict = {}
    #
    # for d in train_set['target']:
    #     for p in d:
    #         if len(p)>=10:
    #             print('%d, %s' %(len(p), ' '.join([idx2word[i] for i in p])))
    #         if len(p) in count_dict:
    #             count_dict[len(p)] += 1
    #         else:
    #             count_dict[len(p)] = 1
    #
    # total_count = sum(count_dict.values())
    #
    # for leng,count in count_dict.items():
    #     print('%s: %d, %.3f' % (leng, count, float(count)/float(total_count)*100))
    #
    # print('Total phrases: %d'% total_count)