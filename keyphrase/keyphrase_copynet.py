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

setup = setup_keyphrase_all # setup_keyphrase_all_testing

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
    '''
    return the copy matrix
    '''
    cc = np.zeros((source.shape[0], target.shape[1], source.shape[1]), dtype='float32')
    for k in xrange(source.shape[0]): # go over each sample in source batch
        for j in xrange(target.shape[1]): # go over each word in target (all target have same length after padding)
            for i in xrange(source.shape[1]): # go over each word in source
                if (source[k, i] == target[k, j]) and (source[k, i] > 0): # if word match, set cc[k][j][i] = 1. Don't count non-word(source[k, i]=0)
                    cc[k][j][i] = 1.
    return cc

def cc_martix_multi_output(source, target):
    '''
    return the copy matrix
    '''
    cc = np.zeros((source.shape[0], target.shape[1], source.shape[1]), dtype='float32')
    for k in range(source.shape[0]): # go over each sample in source batch
        for l in range(target.shape[0]):
            for j in range(target.shape[1]): # go over each word in target (all target have same length after padding)
                for i in range(source.shape[1]): # go over each word in source
                    if (source[k, i] == target[k, j]) and (source[k, i] > 0): # if word match, set cc[k][j][i] = 1. Don't count non-word(source[k, i]=0)
                        cc[k][l][j][i] = 1.
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

def build_data(data):
    # create fuel dataset.
    dataset = datasets.IndexableDataset(indexables=OrderedDict([('source', data['source']),
                                                                ('target', data['target']),
                                                                # ('target_c', data['target_c']),
                                                                ]))
    dataset.example_iteration_scheme \
        = schemes.ShuffledExampleScheme(dataset.num_examples)
    return dataset


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
    logger.info('Start!')

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

    epoch   = 0
    epochs = 10
    valid_best_score    = (float(sys.maxint),float(sys.maxint))
    valids_not_improved = 0
    patience            = 5
    while epoch < epochs:
        epoch += 1
        loss  = []

        # do training?
        do_train     = True
        # do_train   = False
        # do predicting?
        # do_predict = True
        do_predict   = False
        # do testing?
        # do_evaluate  = True
        do_evaluate  = False

        if do_train:
            # train_batches = output_stream(train_data, config['batch_size']).get_epoch_iterator(as_dict=True)

            logger.info('\nEpoch = {} -> Training Set Learning...'.format(epoch))
            progbar = Progbar(train_size / config['batch_size'], logger)

            # number of minibatches
            num_batches = int(float(len(train_data_plain)) / config['batch_size'])
            name_ordering = np.arange(len(train_data_plain), dtype=np.int32)
            np.random.shuffle(name_ordering)
            batch_start = 0

            if config['resume_training'] and epoch == 1:
                name_ordering, batch_start, loss = deserialize_from_file(config['training_archive'])
                batch_start += 1
                # batch_start = 40001

            for batch_id in range(batch_start, num_batches):
                # 1. Prepare data
                data_ids = name_ordering[batch_id * config['batch_size']:min((batch_id + 1) * config['batch_size'], len(train_data_plain))]

                # obtain data
                data_s = train_data_source[data_ids]
                data_t = train_data_target[data_ids]

                # if not multi_output, split one data (with multiple targets) into multiple ones
                if not config['multi_output']:
                    data_s, data_t = split_into_multiple_and_padding(data_s, data_t)
                # validate whether add one unk to the end
                loss_batch = []

                # 2. Training
                #       split into smaller batches, as some samples contains too many outputs, lead to out-of-memory  9195998617
                # for minibatch_id in range(int(math.ceil(len(data_s)/config['mini_batch_size']))):
                #     mini_data_s = data_s[minibatch_id * config['mini_batch_size']:min((minibatch_id + 1) * config['mini_batch_size'], len(data_s))]
                #     mini_data_t = data_t[minibatch_id * config['mini_batch_size']:min((minibatch_id + 1) * config['mini_batch_size'], len(data_t))]
                dd = 0
                max_size = 300000
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

                    if config['copynet']:
                        if config['multi_output']:
                            data_c = cc_martix_multi_output(mini_data_s, mini_data_t)
                        else:
                            data_c = cc_martix(mini_data_s, mini_data_t)
                         # data_c = prepare_batch(batch, 'target_c', data_t.shape[1])
                        loss_batch += [agent.train_(unk_filter(mini_data_s), unk_filter(mini_data_t), data_c)]
                        # loss += [agent.train_guard(unk_filter(mini_data_s), unk_filter(mini_data_t), data_c)]
                    else:
                        loss_batch += [agent.train_(unk_filter(mini_data_s), unk_filter(mini_data_t))]

                    mini_data_s = []
                    mini_data_t = []
                    stack_size  = 0
                mean_ll  = np.average(np.concatenate([l[0] for l in loss_batch]))
                mean_ppl = np.average(np.concatenate([l[1] for l in loss_batch]))
                loss.append([mean_ll, mean_ppl])
                # print progress
                progbar.update(batch_id, [('loss_reg', loss[-1][0]),
                                          ('ppl.', loss[-1][1])])
                # 3. Quick testing
                if batch_id % 10 == 0 and batch_id > 1:
                    print_case = '-' * 100 +'\n'

                    logger.info('Echo={} Evaluation Sampling.'.format(batch_id))
                    print_case += 'Echo={} Evaluation Sampling.\n'.format(batch_id)

                    logger.info('generating [training set] samples')
                    print_case += 'generating [training set] samples\n'

                    for _ in xrange(2):
                        idx              = int(np.floor(n_rng.rand() * train_size))

                        test_s_o, test_t_o = train_data_plain[idx]

                        if not config['multi_output']:
                            # create <abs, phrase> pair for each phrase
                            test_s, test_t = split_into_multiple_and_padding([test_s_o], [test_t_o])

                        inputs_unk = np.asarray(unk_filter(np.asarray(test_s[0], dtype='int32')), dtype='int32')
                        prediction, score = agent.generate_multiple(inputs_unk[None, :], return_all=True)

                        outs, metrics = agent.evaluate_multiple([test_s[0]], [test_t],
                                                                [test_s_o], [test_t_o],
                                                                [prediction], [score],
                                                                idx2word)
                        print '*' * 50

                    logger.info('generating [testing set] samples')
                    for _ in xrange(2):
                        idx            = int(np.floor(n_rng.rand() * test_size))
                        test_s_o, test_t_o = test_data_plain[idx]
                        if not config['multi_output']:
                            test_s, test_t = split_into_multiple_and_padding([test_s_o], [test_t_o])

                        inputs_unk = np.asarray(unk_filter(np.asarray(test_s[0], dtype='int32')), dtype='int32')
                        prediction, score = agent.generate_multiple(inputs_unk[None, :], return_all=True)

                        outs, metrics = agent.evaluate_multiple([test_s[0]], [test_t],
                                                                [test_s_o], [test_t_o],
                                                                [prediction], [score],
                                                                idx2word)
                        print '*' * 50

                    # write examples to log file
                    with open(config['casestudy_log'], 'w+') as print_case_file:
                        print_case_file.write(print_case)
                # 4. Save model
                if batch_id % 10 == 0 and batch_id > 1:
                    # save the weights every K rounds
                    agent.save(config['path_experiment'] + '/experiments.{0}.id={1}.epoch={2}.batch={3}.pkl'.format(config['task_name'], config['timemark'], epoch, batch_id))
                    # save the game(training progress) in case of interrupt!
                    serialize_to_file([name_ordering, batch_id, loss], config['path_experiment'] + '/save_training_status.id={0}.epoch={1}.batch={2}.pkl'.format(config['timemark'], epoch, batch_id))
                    # agent.save_weight_json(config['path_experiment'] + '/weight.print.id={0}.epoch={1}.batch={2}.json'.format(config['timemark'], epoch, batch_id))

                # 5. Evaluate on validation data, and do early-stopping
                if batch_id % 10 == 0 and not (batch_id==0 and epoch==1):
                    logger.info('Validate @ epoch=%d, batch=%d' % (epoch, batch_id))
                    # 1. Prepare data
                    data_s = np.array(validation_set['source'])
                    data_t = np.array(validation_set['target'])
                    # if not multi_output, split one data (with multiple targets) into multiple ones
                    if not config['multi_output']:
                        data_s, data_t = split_into_multiple_and_padding(data_s, data_t)

                    loss_valid = []

                    # for minibatch_id in range(int(math.ceil(len(data_s)/config['mini_batch_size']))):
                    #     mini_data_s = data_s[minibatch_id * config['mini_batch_size']:min((minibatch_id + 1) * config['mini_batch_size'], len(data_s))]
                    #     mini_data_t = data_t[minibatch_id * config['mini_batch_size']:min((minibatch_id + 1) * config['mini_batch_size'], len(data_t))]

                    dd = 0
                    max_size = 300000
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

                        if config['copynet']:
                            data_c = cc_martix(mini_data_s, mini_data_t)
                            loss_valid += [agent.validate_(unk_filter(mini_data_s), unk_filter(mini_data_t), data_c)]
                        else:
                            loss_valid += [agent.validate_(unk_filter(mini_data_s), unk_filter(mini_data_t))]

                        print('\t %d / %d' % (dd, int(math.ceil(len(data_s)/config['mini_batch_size']))))

                        mini_data_s = []
                        mini_data_t = []
                        stack_size = 0

                    logger.info('\tPrevious best score: \t ll=%f, \t ppl=%f' % (valid_best_score[0], valid_best_score[1]))
                    logger.info('\tCurrent score: \t ll=%f, \t ppl=%f' % (np.mean([l[0] for l in loss_valid]), np.mean([l[1] for l in loss_valid])))

                    if np.mean([l[0] for l in loss_valid]) < valid_best_score[0]:
                        valid_best_score = (np.mean([l[0] for l in loss_valid]), np.mean([l[1] for l in loss_valid]))
                        logger.info('New best score')
                        valids_not_improved = 0
                    else:
                        valids_not_improved += 1
                        logger.info('Not improved for %s tests.' % valids_not_improved)

                    if valids_not_improved >= patience:
                        print "Not improved for %s epochs. Stopping..." % patience
                        break

        '''
        test accuracy and f-score at the end of each epoch
        '''
        if do_predict:
            for dataset_name in config['testing_datasets']:
                # override the original test_set
                # test_set = load_testing_data(dataset_name, kwargs=dict(basedir=config['path']))(idx2word, word2idx, config['preprocess_type'])

                test_set = test_sets[dataset_name]

                # print(dataset_name)
                # print('Avg length=%d, Max length=%d' % (np.average([len(s) for s in test_set['source']]), np.max([len(s) for s in test_set['source']])))
                test_data_plain = zip(*(test_set['source'], test_set['target']))
                test_size = len(test_data_plain)

                progbar_test = Progbar(test_size, logger)
                logger.info('Predicting on %s' % dataset_name)

                predictions = []
                scores = []
                test_s_list = []
                test_t_list = []
                test_s_o_list = []
                test_t_o_list = []

                # Predict on testing data
                for idx in xrange(len(test_data_plain)): # len(test_data_plain)
                    test_s_o, test_t_o = test_data_plain[idx]
                    if not config['multi_output']:
                        test_s, test_t = split_into_multiple_and_padding([test_s_o], [test_t_o])
                    test_s = test_s[0]

                    test_s_list.append(test_s)
                    test_t_list.append(test_t)
                    test_s_o_list.append(test_s_o)
                    test_t_o_list.append(test_t_o)

                    inputs_unk = np.asarray(unk_filter(np.asarray(test_s, dtype='int32')), dtype='int32')
                    # print(len(inputs_unk))

                    prediction, score = agent.generate_multiple(inputs_unk[None, :], return_all=True)
                    predictions.append(prediction)
                    scores.append(score)
                    progbar_test.update(idx, [])
                # store predictions in file
                serialize_to_file([test_s_list, test_t_list, test_s_o_list, test_t_o_list, predictions, scores, idx2word], config['predict_path']+'predict.{0}.{1}.pkl'.format(config['predict_type'], dataset_name))

        '''
        Evaluate on Testing Data
        '''
        if do_evaluate:

            for dataset_name in config['testing_datasets']:
                print_test = open(config['predict_path'] + '/experiments.{0}.id={1}.testing@{2}.{3}.len={4}.beam={5}.log'.format(config['task_name'],config['timemark'],dataset_name, config['predict_type'], config['max_len'], config['sample_beam']), 'w')

                test_s_list, test_t_list, test_s_o_list, test_t_o_list, predictions, scores, idx2word = deserialize_from_file(config['predict_path']+'predict.{0}.{1}.pkl'.format(config['predict_type'], dataset_name))

                print_test.write('Testing on %s size=%d @ epoch=%d \n' % (dataset_name, test_size, epoch))
                overall_score = {'p':0.0, 'r':0.0, 'f1':0.0}
                # load from predicted result
                # Evaluation
                outs, metrics = keyphrase_utils.evaluate_multiple(config, test_s_list, test_t_list,
                                                            test_s_o_list, test_t_o_list,
                                                            predictions, scores, idx2word)

                print_test.write(' '.join(outs))
                logger.info('*' * 50)

                real_test_size = sum([1 if m['target_number'] > 0 else 0 for m in metrics])

                for k in [5,10,15]:
                    # Get the Micro Measures
                    overall_score['p@%d' % k] = float(sum([m['p@%d' % k] for m in metrics]))/float(real_test_size)
                    overall_score['r@%d' % k] = float(sum([m['r@%d' % k] for m in metrics]))/float(real_test_size)
                    overall_score['f1@%d' % k] = float(sum([m['f1@%d' % k] for m in metrics]))/float(real_test_size)

                    # Get the Macro Measures
                    correct_number = sum([m['correct_number@%d' % k] for m in metrics])
                    valid_target_number = sum([m['valid_target_number'] for m in metrics])
                    target_number = sum([m['target_number'] for m in metrics])
                    overall_score['macro_p@%d' % k]  = correct_number / float(real_test_size * k)
                    overall_score['macro_r@%d' % k]  = correct_number / float(valid_target_number)
                    if overall_score['macro_p@%d' % k] + overall_score['macro_r@%d' % k] > 0:
                        overall_score['macro_f1@%d' % k] = 2 * overall_score['macro_p@%d' % k] * overall_score['macro_r@%d' % k] / float(overall_score['macro_p@%d' % k] + overall_score['macro_r@%d' % k])
                    else:
                        overall_score['macro_f1@%d' % k] = 0

                    str = 'Overall - %s valid testing data=%d, Number of Target=%d/%d, Number of Prediction=%d, Number of Correct=%d' % (config['predict_type'], real_test_size, valid_target_number, target_number, real_test_size * k, correct_number)
                    logger.info(str)
                    print_test.write(str)
                    str = 'Micro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (k, overall_score['p@%d' % k], k, overall_score['r@%d' % k], k, overall_score['f1@%d' % k])
                    logger.info(str)
                    print_test.write(str)

                    str = 'Macro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (k, overall_score['macro_p@%d' % k], k, overall_score['macro_r@%d' % k], k, overall_score['macro_f1@%d' % k])
                    logger.info(str)
                    print_test.write(str)

                    print('-' * 50)

                logger.info(overall_score)
                print_test.close()

            exit()

            # write examples to log file
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
