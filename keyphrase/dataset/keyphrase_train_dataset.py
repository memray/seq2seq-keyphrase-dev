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
from keyphrase.dataset import data_utils
from keyphrase_test_dataset import DataLoader,load_testing_data
import data_utils as utils


wordfreq = dict()

def load_pairs(records, filter=False):
    global wordfreq
    filtered_records = []
    pairs = []

    for id, record in enumerate(records):
        text        = utils.prepare_text(record, tokenize_sentence = False)
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

    return filtered_records, pairs

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

def load_data_and_dict(training_dataset):
    '''
    here dict is built on both training and testing dataset, which may be not suitable (testing data should be unseen)
    :param training_dataset,testing_dataset: path
    :return:
    '''
    global wordfreq

    # load training dataset
    print('Loading training dataset')
    f                   = open(training_dataset, 'r')
    training_records    = json.load(f)
    title_dict          = dict([(r['title'].strip().lower(), r) for r in training_records])
    print('#(Training Data)=%d' % len(title_dict))

    # load testing dataset
    print('Loading testing dataset')
    testing_names       = config['testing_datasets'] # only these three may have overlaps with training data
    testing_records     = {}

    # rule out the ones appear in testing data
    print('Filtering testing dataset from training data')
    for dataset_name in testing_names:
        print(dataset_name)

        testing_records[dataset_name] = load_testing_data(dataset_name, kwargs=dict(basedir = config['path'])).get_docs()

        for r in testing_records[dataset_name]:
            title = r['title'].strip().lower()
            if title in title_dict:
                title_dict.pop(title)


    print('Process the data')
    training_records, train_pairs         = load_pairs(title_dict.values(), filter=True)
    print('#(Training Data after Filtering Noises)=%d' % len(training_records))

    print('Preparing development data')
    training_records    = numpy.asarray(training_records)
    train_pairs         = numpy.asarray(train_pairs)
    validation_ids      = numpy.random.randint(0, len(training_records), config['validation_size'])
    # keep a copy of validation data
    if 'validation_id' in config and os.path.exists(config['validation_id']):
        validation_ids = deserialize_from_file(config['validation_id'])
    else:
        serialize_to_file(validation_ids, config['validation_id'])

    validation_records  = training_records[validation_ids]
    validation_pairs    = train_pairs[validation_ids]
    training_records    = numpy.delete(training_records, validation_ids, axis=0)
    train_pairs         = numpy.delete(train_pairs, validation_ids, axis=0)

    print('#(Training Data after Filtering Validate & Test data)=%d' % len(train_pairs))

    print('Preparing testing data KE20k')
    testing_ids         = numpy.random.randint(0, len(training_records), config['validation_size'])
    # keep a copy of testing data
    if 'testing_id' in config and os.path.exists(config['testing_id']):
        testing_ids = deserialize_from_file(config['testing_id'])
    else:
        serialize_to_file(testing_ids, config['testing_id'])
    testing_records['ke20k']  = training_records[testing_ids]
    training_records          = numpy.delete(training_records, testing_ids, axis=0)
    train_pairs               = numpy.delete(train_pairs, testing_ids, axis=0)


    test_pairs                = dict([(k, load_pairs(v, filter=False)[1]) for (k,v) in testing_records.items()])

    print('Building dicts')
    # if voc exists and is assigned, load it, overwrite the wordfreq
    if 'voc' in config:
        print('Loading dicts from %s' % config['voc'])
        wordfreq = dict(deserialize_from_file(config['voc']))
    idx2word, word2idx = build_dict(wordfreq)

    # use character-based model [on]
    # use word-based model     [off]
    print('Mapping tokens to indexes')
    train_set           = data_utils.build_data(train_pairs, idx2word, word2idx)
    validation_set      = data_utils.build_data(validation_pairs, idx2word, word2idx)
    test_set            = dict([(k, data_utils.build_data(v, idx2word, word2idx)) for (k,v) in test_pairs.items()])

    print('Train samples      : %d' % len(train_pairs))
    print('Validation samples : %d' % len(validation_pairs))
    print('Test samples       : %d' % sum([len(test_pair) for test_pair in test_pairs.values()]))
    print('Dict size          : %d' % len(idx2word))

    return train_set, validation_set, test_set, idx2word, word2idx

if __name__ == '__main__':
    # config = config.setup_keyphrase_all()
    config = setup_keyphrase_all()

    start_time = time.clock()
    train_set, validation_set, test_set, idx2word, word2idx = load_data_and_dict(config['training_dataset'])
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