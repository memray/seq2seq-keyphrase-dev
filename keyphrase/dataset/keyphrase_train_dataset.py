# coding=utf-8
import json
import sys
import time

import nltk
import numpy
import numpy as np
import re

from keyphrase.config import *
from emolga.dataset.build_dataset import *
from keyphrase_test_dataset import DataLoader,load_testing_data
import data_utils as utils


wordfreq = dict()

def load_pairs(records):
    global wordfreq
    pairs = []

    for id, record in enumerate(records):
        text        = utils.prepare_text(record)
        tokens      = utils.get_tokens(text)
        keyphrases  = utils.process_keyphrase(record)

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

        pairs.append((tokens, keyphrases))
        if id % 10000 == 0:
            print('%d \n\t%s \n\t%s \n\t%s' % (id, text, tokens, keyphrases))

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

def load_data_and_dict(training_dataset, testing_dataset):
    '''
    here dict is built on both training and testing dataset, which may be not suitable (testing data should be unseen)
    :param training_dataset,testing_dataset: path
    :return:
    '''
    global wordfreq

    # load training dataset
    f                   = open(training_dataset, 'r')
    training_records    = json.load(f)
    title_dict          = dict([(r['title'].lower(), r) for r in training_records])

    # load testing dataset
    testing_names       = ['inspec','nus','semeval'] # only these three may have overlaps with training data
    testing_records     = {}

    # rule out the ones appear in testing data
    for dataset_name in testing_names:
        testing_records[dataset_name] = load_testing_data(dataset_name, kwargs=dict(basedir = config['path'])).get_docs()

        for r in testing_records[dataset_name]:
            title = r['title'].strip().lower()
            if title in title_dict:
                title_dict.pop(title)

    training_records    = numpy.asarray(title_dict.values())
    validation_ids      = numpy.random.randint(0, len(training_records), 1000)

    # keep a copy of validation data
    if 'validation_id' in config and os.path.exists(config['validation_id']):
        validation_ids = deserialize_from_file(config['validation_id'])
    else:
        serialize_to_file(validation_ids, config['validation_id'])

    validation_records  = training_records[validation_ids]
    training_records    = numpy.delete(training_records, validation_ids)

    train_pairs         = load_pairs(training_records)
    validation_pairs    = load_pairs(validation_records)
    test_pairs          = dict([(k, load_pairs(v)) for (k,v) in testing_records.items()])

    print('read dataset done.')

    # if voc exists and is assigned, load it, overwrite the wordfreq
    if 'voc' in config:
        wordfreq = dict(deserialize_from_file(config['voc']))

    idx2word, word2idx = build_dict(wordfreq)
    print('build dicts done.')

    # use character-based model [on]
    # use word-based model     [off]
    train_set           = build_data(train_pairs, idx2word, word2idx)
    validation_set      = build_data(validation_pairs, idx2word, word2idx)
    test_set            = dict([(k, build_data(v, idx2word, word2idx)) for (k,v) in test_pairs.items()])

    print('Train pairs      : %d' % len(train_pairs))
    print('Validation pairs : %d' % len(validation_pairs))
    print('Test pairs       : %d' % sum([len(test_pair) for test_pair in test_pairs.values()]))
    print('Dict size        : %d' % len(idx2word))

    return train_set, validation_set, test_set, idx2word, word2idx

if __name__ == '__main__':
    # config = config.setup_keyphrase_all()
    config = setup_keyphrase_all()

    start_time = time.clock()
    train_set, validation_set, test_set, idx2word, word2idx = load_data_and_dict(config['training_dataset'], config['testing_dataset'])
    serialize_to_file([train_set, validation_set, test_set, idx2word, word2idx], config['dataset'])
    print('Finish processing and dumping: %d seconds' % (time.clock()-start_time))
    #
    # # export vocabulary to file for manual check
    # wordfreq = sorted(wordfreq.items(), key=lambda a: a[1], reverse=True)
    # serialize_to_file(wordfreq, config['voc'])


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