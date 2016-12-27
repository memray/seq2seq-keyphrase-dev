"""
This is the implementation of Copy-NET
We start from the basic Seq2seq framework for a auto-encoder.
"""
import logging
import time
import numpy as np
import sys
import copy

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from keyphrase import keyphrase_dataset
from keyphrase.config import *
from emolga.utils.generic_utils import *
from emolga.models.covc_encdec import NRM
from emolga.models.encdec import NRM as NRM0
from emolga.dataset.build_dataset import deserialize_from_file
from collections import OrderedDict
from fuel import datasets
from fuel import transformers
from fuel import schemes

setup = setup_keyphrase_all # setup_keyphrase_acm

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
    # ch = logging.StreamHandler(sys.stdout)

    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # fh.setLevel(logging.INFO)
    # ch.setLevel(logging.INFO)
    # logging.getLogger().addHandler(ch)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    return logging

# prepare logging.
config  = setup()   # load settings.
# for w in config:
#     print '{0}={1}'.format(w, config[w])

print('Log path: %s' % (config['path_experiment'] + '/experiments.{0}.id={1}.log'.format(config['task_name'],config['timemark'])))
logger  = init_logging(config['path_experiment'] + '/experiments.{0}.id={1}.log'.format(config['task_name'],config['timemark']))

# log = logging.getLogger()
# sys.stdout = LoggerWriter(log.debug)
# sys.stderr = LoggerWriter(log.warning)

n_rng   = np.random.RandomState(config['seed'])
np.random.seed(config['seed'])
rng     = RandomStreams(n_rng.randint(2 ** 30))
logger.info('Start!')

train_set, test_set, idx2word, word2idx = deserialize_from_file(config['dataset'])
logger.info('Load data done.')
# data is too large to dump into file, so load from raw dataset directly
# train_set, test_set, idx2word, word2idx = keyphrase_dataset.load_data_and_dict(config['training_dataset'], config['testing_dataset'])

if config['voc_size'] == -1:   # not use unk
    config['enc_voc_size'] = max(zip(*word2idx.items())[1]) + 1
    config['dec_voc_size'] = config['enc_voc_size']
else:
    config['enc_voc_size'] = config['voc_size']
    config['dec_voc_size'] = config['enc_voc_size']

samples  = len(train_set['source'])

logger.info('build dataset done. ' +
            'dataset size: {} ||'.format(samples) +
            'vocabulary size = {0}/ batch size = {1}'.format(
        config['dec_voc_size'], config['batch_size']))


def build_data(data):
    # create fuel dataset.
    dataset     = datasets.IndexableDataset(indexables=OrderedDict([('source', data['source']),
                                                                    ('target', data['target']),
                                                                    # ('target_c', data['target_c']),
                                                                    ]))
    dataset.example_iteration_scheme \
                = schemes.ShuffledExampleScheme(dataset.num_examples)
    return dataset


train_data        = build_data(train_set) # a fuel IndexableDataset
train_data_plain  = zip(*(train_set['source'], train_set['target']))
test_data_plain   = zip(*(test_set['source'],  test_set['target']))

train_size        = len(train_data_plain)
test_size         = len(test_data_plain)
tr_idx            = n_rng.permutation(train_size)[:2000].tolist()
ts_idx            = n_rng.permutation(test_size )[:2000].tolist()
logger.info('load the data ok.')

# build the agent
if config['copynet']:
    agent  = NRM(config, n_rng, rng, mode=config['mode'],
                 use_attention=True, copynet=config['copynet'], identity=config['identity'])
else:
    agent  = NRM0(config, n_rng, rng, mode=config['mode'],
                  use_attention=True, copynet=config['copynet'], identity=config['identity'])

agent.build_()
agent.compile_('all')
logger.info('compile ok.')

# load pre-trained model
if config['trained_model']:
    logger.info('Trained model exists, loading from %s' % config['trained_model'])
    agent.load(config['trained_model'])

epoch   = 0
epochs = 10
while epoch < epochs:
    epoch += 1
    loss  = []

    def output_stream(dataset, batch_size, size=1):
        data_stream = dataset.get_example_stream()
        data_stream = transformers.Batch(data_stream,
                                         iteration_scheme=schemes.ConstantScheme(batch_size))

        # add padding and masks to the dataset
        # Warning: in multiple output case, will raise ValueError: All dimensions except length must be equal, need padding manually
        # data_stream = transformers.Padding(data_stream, mask_sources=('source', 'target', 'target_c'))
        # data_stream = transformers.Padding(data_stream, mask_sources=('source', 'target'))
        return data_stream

    def prepare_batch(batch, mask, fix_len=None):
        data = batch[mask].astype('int32')
        data = np.concatenate([data, np.zeros((data.shape[0], 1), dtype='int32')], axis=1)

        def cut_zeros(data, fix_len=None):
            if fix_len is not None:
                return data[:, : fix_len]
            for k in range(data.shape[1] - 1, 0, -1):
                data_col = data[:, k].sum()
                if data_col > 0:
                    return data[:, : k + 2]
            return data
        data = cut_zeros(data, fix_len)
        return data

    def cc_martix(source, target):
        cc = np.zeros((source.shape[0], target.shape[1], source.shape[1]), dtype='float32')
        for k in xrange(source.shape[0]):
            for j in xrange(target.shape[1]):
                for i in xrange(source.shape[1]):
                    if (source[k, i] == target[k, j]) and (source[k, i] > 0):
                        cc[k][j][i] = 1.
        return cc

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
            # mask shows whether each word is frequent or not, only word_index<config['voc_size']=1, else=0
            mask = (np.less(data, config['voc_size'])).astype(dtype='int32')
            # low frequency word will be set to 1 (index of <unk>)
            data = copy.copy(data * mask + (1 - mask))
            return data

    def split_into_multiple(data_s_o, data_t_o):
        data_s = []
        data_t = []
        for s,t in zip(data_s_o, data_t_o):
            for p in t:
                data_s += [s]
                data_t += [p]
        return data_s, data_t

    # training
    notrain = False
    if not notrain:
        train_batches = output_stream(train_data, config['batch_size']).get_epoch_iterator(as_dict=True)

        logger.info('\nEpoch = {} -> Training Set Learning...'.format(epoch))
        progbar = Progbar(train_size / config['batch_size'], logger)
        for it, batch in enumerate(train_batches):
            print(it)
            # obtain data
            data_s = prepare_batch(batch, 'source')
            data_t = prepare_batch(batch, 'target')

            # if not multi_output, split one data (with multiple targets) into multiple ones
            if not config['multi_output']:
                data_s, data_t = split_into_multiple(data_s, data_t)

            if config['copynet']:
                data_c = cc_martix(data_s, data_t)
                # data_c = prepare_batch(batch, 'target_c', data_t.shape[1])
                loss += [agent.train_(unk_filter(data_s), unk_filter(data_t), data_c)]
            else:
                loss += [agent.train_(unk_filter(data_s), unk_filter(data_t))]

            progbar.update(it, [('loss_reg', loss[-1][0]), ('ppl.', loss[-1][1])])

            if it % 1000 == 0:
                logger.info('Echo={} Evaluation Sampling.'.format(it))
                logger.info('generating [training set] samples')
                for _ in xrange(5):
                    idx              = int(np.floor(n_rng.rand() * train_size))
                    train_s, train_t = train_data_plain[idx]

                    if not config['multi_output']:
                        train_s, train_t = split_into_multiple(train_s, train_t)

                    v                = agent.evaluate_(np.asarray(train_s, dtype='int32'),
                                                       np.asarray(train_t, dtype='int32'),
                                                       idx2word,
                                                       np.asarray(unk_filter(train_s), dtype='int32'))
                    print '*' * 50

                logger.info('generating [testing set] samples')
                for _ in xrange(5):
                    idx            = int(np.floor(n_rng.rand() * test_size))
                    test_s, test_t = test_data_plain[idx]
                    test_s, test_t = split_into_multiple(test_s, test_t)
                    v              = agent.evaluate_(np.asarray(test_s, dtype='int32'),
                                                     np.asarray(test_t, dtype='int32'),
                                                     idx2word,
                                                     np.asarray(unk_filter(test_s), dtype='int32'))
                    print '*' * 50

            # save the weights every K rounds
            if it % 1000 == 0:
                agent.save(config['path_experiment'] + '/experiments.{0}.id={1}.epoch={2}.pkl'.format(config['task_name'], config['timemark'], epoch))

    # # test accuracy
    # progbar_tr = Progbar(2000)
    #
    # print '\n' + '__' * 50
    # gen, gen_pos = 0, 0
    # cpy, cpy_pos = 0, 0
    # for it, idx in enumerate(tr_idx):
    #     train_s, train_t = train_data_plain[idx]
    #
    #     c = agent.analyse_(np.asarray(train_s, dtype='int32'),
    #                        np.asarray(train_t, dtype='int32'),
    #                        idx2word)
    #     if c[1] == 0:
    #         # generation mode
    #         gen     += 1
    #         gen_pos += c[0]
    #     else:
    #         # copy mode
    #         cpy     += 1
    #         cpy_pos += c[0]
    #
    #     progbar_tr.update(it + 1, [('Gen', gen_pos), ('Copy', cpy_pos)])
    #
    # logger.info('\nTraining Accuracy:' +
    #             '\tGene-Mode: {0}/{1} = {2}%'.format(gen_pos, gen, 100 * gen_pos/float(gen)) +
    #             '\tCopy-Mode: {0}/{1} = {2}%'.format(cpy_pos, cpy, 100 * cpy_pos/float(cpy)))
    #
    # progbar_ts = Progbar(2000)
    # print '\n' + '__' * 50
    # gen, gen_pos = 0, 0
    # cpy, cpy_pos = 0, 0
    # for it, idx in enumerate(ts_idx):
    #     test_s, test_t = test_data_plain[idx]
    #     c      = agent.analyse_(np.asarray(test_s, dtype='int32'),
    #                             np.asarray(test_t, dtype='int32'),
    #                             idx2word)
    #     if c[1] == 0:
    #         # generation mode
    #         gen     += 1
    #         gen_pos += c[0]
    #     else:
    #         # copy mode
    #         cpy     += 1
    #         cpy_pos += c[0]
    #
    #     progbar_ts.update(it + 1, [('Gen', gen_pos), ('Copy', cpy_pos)])
    #
    # logger.info('\nTesting Accuracy:' +
    #             '\tGene-Mode: {0}/{1} = {2}%'.format(gen_pos, gen, 100 * gen_pos/float(gen)) +
    #             '\tCopy-Mode: {0}/{1} = {2}%'.format(cpy_pos, cpy, 100 * cpy_pos/float(cpy)))
