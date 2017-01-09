"""
This is the implementation of Copy-NET
We start from the basic Seq2seq framework for a auto-encoder.
"""
import logging
import time
import numpy as np
import sys
import copy
import math

import theano

import keyphrase_utils

theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from keyphrase.dataset.keyphrase_train_dataset import *
from keyphrase.config import *
from emolga.utils.generic_utils import *
from emolga.models.covc_encdec import NRM
from emolga.models.encdec import NRM as NRM0
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
from collections import OrderedDict
from fuel import datasets
from fuel import transformers
from fuel import schemes

setup = setup_keyphrase_all_testing # setup_keyphrase_all_testing

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def init_logging(logfile):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S'   )
    fh = logging.FileHandler(logfile)
    # ch = logging.StreamHandler()
    ch = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)
    logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging

def unk_filter(data):
    '''
    only keep the top [voc_size] frequent words, replace the other as 0
    word index is in the order of from most frequent to least
    :param data:
    :return:
    '''
    if config['voc_size'] == -1:
        return copy.copy(data)
    else:
        # mask shows whether keeps each word (frequent) or not, only word_index<config['voc_size']=1, else=0
        mask = (np.less(data, config['voc_size'])).astype(dtype='int32')
        # low frequency word will be set to 1 (index of <unk>)
        data = copy.copy(data * mask + (1 - mask))
        return data

def add_padding(data):
    shapes = [np.asarray(sample).shape for sample in data]
    lengths = [shape[0] for shape in shapes]

    # make sure there's at least one zero at last to indicate the end of sentence <eol>
    max_sequence_length = max(lengths) + 1
    rest_shape = shapes[0][1:]
    padded_batch = np.zeros(
        (len(data), max_sequence_length) + rest_shape,
        dtype='int32')
    for i, sample in enumerate(data):
        padded_batch[i, :len(sample)] = sample

    return padded_batch

def split_into_multiple_and_padding(data_s_o, data_t_o):
    data_s = []
    data_t = []
    for s, t in zip(data_s_o, data_t_o):
        for p in t:
            data_s += [s]
            data_t += [p]

    data_s = add_padding(data_s)
    data_t = add_padding(data_t)
    return data_s, data_t

if __name__ == '__main__':

    # prepare logging.
    config  = setup()   # load settings.

    print('Log path: %s' % (config['path_experiment'] + '/experiments.{0}.id={1}.log'.format(config['task_name'],config['timemark'])))
    logger  = init_logging(config['path_experiment'] + '/experiments.{0}.id={1}.log'.format(config['task_name'],config['timemark']))

    # log = logging.getLogger()
    # sys.stdout = LoggerWriter(log.debug)
    # sys.stderr = LoggerWriter(log.warning)

    n_rng   = np.random.RandomState(config['seed'])
    np.random.seed(config['seed'])
    rng     = RandomStreams(n_rng.randint(2 ** 30))

    logger.info('*'*20 + '  config information  ' + '*'*20)
    # print config information
    for k,v in config.items():
        logger.info("\t\t\t\t%s : %s" % (k,v))
    logger.info('*' * 50)

    train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    # test_set = load_additional_testing_data(config['path']+'/dataset/keyphrase/ir-books/expert-conflict-free.json', idx2word, word2idx)

    logger.info('Load data done.')
    # data is too large to dump into file, so load from raw dataset directly
    # train_set, test_set, idx2word, word2idx = keyphrase_dataset.load_data_and_dict(config['training_dataset'], config['testing_dataset'])

    if config['voc_size'] == -1:   # not use unk
        config['enc_voc_size'] = max(zip(*word2idx.items())[1]) + 1
        config['dec_voc_size'] = config['enc_voc_size']
    else:
        config['enc_voc_size'] = config['voc_size']
        config['dec_voc_size'] = config['enc_voc_size']

    predictions  = len(train_set['source'])

    logger.info('build dataset done. ' +
                'dataset size: {} ||'.format(predictions) +
                'vocabulary size = {0}/ batch size = {1}'.format(
            config['dec_voc_size'], config['batch_size']))

    # train_data        = build_data(train_set) # a fuel IndexableDataset
    train_data_plain  = zip(*(train_set['source'], train_set['target']))
    train_data_source = np.array(train_set['source'])
    train_data_target = np.array(train_set['target'])

    # test_data_plain   = zip(*(test_set['source'],  test_set['target']))

    # trunk the over-long input in testing data
    for test_set in test_sets.values():
        test_set['source'] = [s if len(s)<1000 else s[:1000] for s in test_set['source']]
    test_data_plain = np.concatenate([zip(*(t['source'],  t['target'])) for t in test_sets.values()])

    print('Avg length=%d, Max length=%d' % (
    np.average([len(s[0]) for s in test_data_plain]), np.max([len(s[0]) for s in test_data_plain])))

    train_size        = len(train_data_plain)
    test_size         = len(test_data_plain)
    tr_idx            = n_rng.permutation(train_size)[:2000].tolist()
    ts_idx            = n_rng.permutation(test_size )[:2000].tolist()
    logger.info('load the data ok.')

    # build the agent
    if config['copynet']:
        agent = NRM(config, n_rng, rng, mode=config['mode'],
                     use_attention=True, copynet=config['copynet'], identity=config['identity'])
    else:
        agent = NRM0(config, n_rng, rng, mode=config['mode'],
                      use_attention=True, copynet=config['copynet'], identity=config['identity'])

    agent.build_()
    agent.compile_('all')
    logger.info('compile ok.')

    # load pre-trained model
    if config['trained_model']:
        logger.info('Trained model exists, loading from %s' % config['trained_model'])
        agent.load(config['trained_model'])
        # agent.save_weight_json(config['weight_json'])

    # number of minibatches
    num_batches = int(float(len(train_data_plain)) / config['batch_size'])
    name_ordering = np.arange(len(train_data_plain), dtype=np.int32)
    np.random.shuffle(name_ordering)
    batch_start = 0

    for batch_id in range(batch_start, num_batches):
        # 1. Prepare data
        data_ids = name_ordering[batch_id * config['batch_size']:min((batch_id + 1) * config['batch_size'], len(train_data_plain))]

        # obtain data
        data_s = train_data_source[data_ids]
        data_t = train_data_target[data_ids]

        data_s, data_t = split_into_multiple_and_padding(data_s, data_t)

        # 2. Training
        #       split into smaller batches, as some samples contains too many outputs, lead to out-of-memory  9195998617
        # for minibatch_id in range(int(math.ceil(len(data_s)/config['mini_batch_size']))):
        #     mini_data_s = data_s[minibatch_id * config['mini_batch_size']:min((minibatch_id + 1) * config['mini_batch_size'], len(data_s))]
        #     mini_data_t = data_t[minibatch_id * config['mini_batch_size']:min((minibatch_id + 1) * config['mini_batch_size'], len(data_t))]

        loss_batch = []
        dd = 0
        max_size = 100000
        stack_size = 0
        mini_data_s = []
        mini_data_t = []
        while dd < len(data_s):
            while dd < len(data_s) and stack_size + len(data_s[dd]) * len(data_t[dd]) < max_size:
                mini_data_s.append(data_s[dd])
                mini_data_t.append(data_t[dd])
                stack_size += len(data_s[dd]) * len(data_t[dd])
                dd += 1
            mini_data_s = np.asarray(mini_data_s)
            mini_data_t = np.asarray(mini_data_t)

            # get the generating probability and decoding of targets
            probs, stats = agent.represent_(unk_filter(mini_data_s), unk_filter(mini_data_t))

            for prob, stat, t in zip(probs, stats, mini_data_t):
                lt_voc = [1 if w >= config['voc_size'] else 0 for w in t]

                if sum(lt_voc) > 0:
                    continue

                target = keyphrase_utils.cut_zero(t, idx2word)

                last_word = t[len(target)-1]

                print('Target phrase: [prob=%f][score=%f]%s' % (prob[last_word], - np.log(prob[last_word] + 1e-10), ' '.join(target)))
                print('\t vector: %s' % str(stat))

            mini_data_s = []
            mini_data_t = []
            stack_size  = 0
            print(len(loss_batch))

        # 3. Quick testinge_file

        # 4. Save model

        # 5. Evaluate on validation data, and do early-stopping
