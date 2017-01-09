import math

import logging
from nltk.stem.porter import *
import numpy as np

from dataset import dataset_utils

logger = logging.getLogger(__name__)


def cut_zero(sample, idx2word):
    sample = list(sample)
    if 0 not in sample:
        return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample]
    # return the string before 0 (<eol>)
    return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample[:sample.index(0)]]


def evaluate_multiple(config, test_set, inputs, outputs,
                      original_input, original_outputs,
                      samples, scores, idx2word, do_stem):
    '''
    inputs_unk is same as inputs except for filtered out all the low-freq words to 1 (<unk>)
    return the top few keywords, number is set in config
    :param: original_input, same as inputs, the vector of one input sentence
    :param: original_outputs, vectors of corresponding multiple outputs (e.g. keyphrases)
    :return:
    '''

    # Generate keyphrases
    # if inputs_unk is None:
    #     samples, scores = self.generate_multiple(inputs[None, :], return_all=True)
    # else:
    #     samples, scores = self.generate_multiple(inputs_unk[None, :], return_all=True)

    stemmer = PorterStemmer()
    # Evaluation part
    outs = []
    micro_metrics = []
    micro_matches = []

    # load stopword
    with open(config['path'] + '/dataset/stopword/stopword_en.txt') as stopword_file:
        stopword_set = set([stemmer.stem(w.strip()) for w in stopword_file])

    postag_lists = [[s[1] for s in d] for d in test_set['tagged_source']]
    # for input_sentence, target_list, predict_list, score_list in zip(inputs, original_outputs, samples, scores):
    for source_str, input_sentence, target_list, predict_list, score_list, postag_list in zip(test_set['source_str'], inputs, test_set['target_str'], samples, scores, postag_lists):

        '''
        enumerate each document, process target/predict/score and measure via p/r/f1
        '''
        target_outputs = []
        predict_outputs = []
        predict_scores = []
        predict_set = set()
        correctly_matched = np.asarray([0] * max(len(target_list), len(predict_list)), dtype='int32')

        # stem the original input
        stemmed_input = [stemmer.stem(w) for w in cut_zero(input_sentence, idx2word)]

        # convert target index into string
        for target in target_list:
            # target = cut_zero(target, idx2word)
            if do_stem:
                target = [stemmer.stem(w) for w in target]
            print(target)

            keep = True
            # whether do filtering on groundtruth phrases. if config['target_filter']==None, do nothing
            if config['target_filter']:
                match = None
                for i in range(len(stemmed_input) - len(target) + 1):
                    match = None
                    for j in range(len(target)):
                        if target[j] != stemmed_input[i + j]:
                            match = False
                            break
                    if j == len(target) - 1 and match == None:
                        match = True
                        break

                if match == True:
                    # if match and 'appear-only', keep this phrase
                    if config['target_filter'] == 'appear-only':
                        keep = keep and True
                    elif config['target_filter'] == 'non-appear-only':
                        keep = keep and False
                elif match == False:
                    # if not match and 'appear-only', discard this phrase
                    if config['target_filter'] == 'appear-only':
                        keep = keep and False
                    # if not match and 'non-appear-only', keep this phrase
                    elif config['target_filter'] == 'non-appear-only':
                        keep = keep and True

            if not keep:
                continue

            target_outputs.append(target)

        # check if prediction is noun-phrase, initialize a filter. Be sure this should be after stemming
        if config['noun_phrase_only']:
            stemmed_source = [stemmer.stem(w) for w in source_str]
            noun_phrases = dataset_utils.get_none_phrases(stemmed_source, postag_list, config['max_len'])
            noun_phrase_set = set([' '.join(p[0]) for p in noun_phrases])

        # convert predict index into string
        for id, (predict, score) in enumerate(zip(predict_list, score_list)):
            predict = cut_zero(predict, idx2word)
            predict = [stemmer.stem(w) for w in predict]

            # filter some not good ones
            keep = True
            if len(predict) == 0:
                keep = False
            number_digit = 0
            for w in predict:
                w = w.strip()
                if w == '<unk>' or w == '<eos>':
                    keep = False
                if re.match(r'[_,\(\)\.\'%]', w):
                    keep = False
                    # print('\t\tPunctuations! - %s' % str(predict))
                if w == '<digit>':
                    number_digit += 1

            if len(predict) >= 1 and (predict[0] in stopword_set or predict[-1] in stopword_set):
                keep = False

            if len(predict) <= 1:
                keep = False

            # whether do filtering on predicted phrases. if config['predict_filter']==None, do nothing
            if config['predict_filter']:
                match = None
                for i in range(len(stemmed_input) - len(predict) + 1):
                    match = None
                    for j in range(len(predict)):
                        if predict[j] != stemmed_input[i + j]:
                            match = False
                            break
                    if j == len(predict) - 1 and match == None:
                        match = True
                        break

                if match == True:
                    # if match and 'appear-only', keep this phrase
                    if config['predict_filter'] == 'appear-only':
                        keep = keep and True
                    elif config['predict_filter'] == 'non-appear-only':
                        keep = keep and False
                elif match == False:
                    # if not match and 'appear-only', discard this phrase
                    if config['predict_filter'] == 'appear-only':
                        keep = keep and False
                    # if not match and 'non-appear-only', keep this phrase
                    elif config['predict_filter'] == 'non-appear-only':
                        keep = keep and True

            # if all are <digit>, discard
            if number_digit == len(predict):
                keep = False

            # remove duplicates
            key = '-'.join(predict)
            if key in predict_set:
                keep = False

            # if #(word) == #(letter), it predicts like this: h a s k e l
            if sum([len(w) for w in predict])==len(predict) and len(predict) > 2:
                keep = False
                # print('\t\tall letters! - %s' % str(predict))

            # check if prediction is noun-phrase
            if config['noun_phrase_only']:
                if ' '.join(predict) not in noun_phrase_set:
                    print('Not a NP: %s' % (' '.join(predict)))
                    keep = False

            # discard invalid ones
            if not keep:
                continue

            predict_outputs.append(predict)
            predict_scores.append(score)
            predict_set.add(key)

        # whether keep the longest phrases only, as there're too many phrases are part of other longer phrases
        if config['keep_longest']:
            match_phrase_index = []

            for ii, p_ii in enumerate(predict_outputs): # shorter one
                for jj, p_jj in enumerate(predict_outputs): # longer one
                    if ii==jj or len(p_ii)>=len(p_jj): # p_jj must be longer than p_ii
                        continue

                    match = None
                    for start in range(len(p_jj) - len(p_ii) + 1): # iterate the start of long phrase
                        match = None
                        for w_index in range(len(p_ii)): # iterate the short phrase
                            if (p_ii[w_index]!=p_jj[start+w_index]):
                                match = False
                                break
                        if w_index == len(p_ii) - 1 and match == None:
                            match = True
                            break
                    if match: # p_ii is part of p_jj, discard
                        match_phrase_index.append(ii)
                        # print("Matched pair: %s \t - \t %s" % (str(p_ii), str(p_jj)))
                        # pass
                        break

            predict_outputs = np.delete(predict_outputs, match_phrase_index)
            predict_scores  = np.delete(predict_scores, match_phrase_index)

        # check whether the predicted phrase is correct (match any groundtruth)
        for p_id, predict in enumerate(predict_outputs):
            for target in target_outputs:
                if len(target) == len(predict):
                    flag = True
                    for i, w in enumerate(predict):
                        if predict[i] != target[i]:
                            flag = False
                    if flag:
                        correctly_matched[p_id] = 1
                        # print('%s correct!!!' % predict)


        predict_outputs = np.asarray(predict_outputs)
        predict_scores = np.asarray(predict_scores)
        # normalize the score?
        if config['normalize_score']:
            predict_scores = np.asarray(
                [math.log(math.exp(score) / len(predict)) for predict, score in zip(predict_outputs, predict_scores)])
            score_list_index = np.argsort(predict_scores)
            predict_outputs = predict_outputs[score_list_index]
            predict_scores = predict_scores[score_list_index]
            correctly_matched = correctly_matched[score_list_index]

        metric_dict = {}

        '''
        Compute micro metrics
        '''
        for number_to_predict in [5, 10, 15]:
            metric_dict['appear_target_number'] = len(target_outputs)
            metric_dict['target_number'] = len(target_list)
            metric_dict['correct_number@%d' % number_to_predict] = sum(correctly_matched[:number_to_predict])

            metric_dict['p@%d' % number_to_predict] = float(sum(correctly_matched[:number_to_predict])) / float(
                number_to_predict)

            if len(target_outputs) != 0:
                metric_dict['r@%d' % number_to_predict] = float(sum(correctly_matched[:number_to_predict])) / float(
                    len(target_outputs))
            else:
                metric_dict['r@%d' % number_to_predict] = 0

            if metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict] != 0:
                metric_dict['f1@%d' % number_to_predict] = 2 * metric_dict['p@%d' % number_to_predict] * metric_dict[
                    'r@%d' % number_to_predict] / float(
                    metric_dict['p@%d' % number_to_predict] + metric_dict['r@%d' % number_to_predict])
            else:
                metric_dict['f1@%d' % number_to_predict] = 0

            # Compute the binary preference measure (Bpref)
            bpref = 0.
            trunked_match = correctly_matched[:number_to_predict].tolist()  # get the first K prediction to evaluate
            match_indexes = np.nonzero(trunked_match)[0]

            if len(match_indexes) > 0:
                for mid, mindex in enumerate(match_indexes):
                    bpref += 1. - float(mindex - mid) / float(number_to_predict)  # there're mindex elements, and mid elements are correct, before the (mindex+1)-th element
                metric_dict['bpref@%d' % number_to_predict] = float(bpref)/float(len(match_indexes))
            else:
                metric_dict['bpref@%d' % number_to_predict] = 0

            # Compute the mean reciprocal rank (MRR)
            rank_first = 0
            try:
                rank_first = trunked_match.index(1) + 1
            except ValueError:
                pass

            if rank_first > 0:
                metric_dict['mrr@%d' % number_to_predict] = float(1)/float(rank_first)
            else:
                metric_dict['mrr@%d' % number_to_predict] = 0

        micro_metrics.append(metric_dict)
        micro_matches.append(correctly_matched)

        '''
        Print information on each prediction
        '''
        # print stuff
        a = '[SOURCE][{0}]: {1}'.format(len(input_sentence) ,' '.join(cut_zero(input_sentence, idx2word)))
        logger.info(a)
        a += '\n'

        b = '[TARGET]: %d/%d targets\n\t\t' % (len(target_outputs), len(target_list))
        for id, target in enumerate(target_outputs):
            b += ' '.join(target) + '; '
        logger.info(b)
        b += '\n'
        c = '[DECODE]: %d/%d predictions' % (len(predict_outputs), len(predict_list))
        for id, (predict, score) in enumerate(zip(predict_outputs, predict_scores)):
            c += ('\n\t\t[%.3f][%d][%d]' % (score, len(predict), sum([len(w) for w in predict]))) + ' '.join(predict)
            if correctly_matched[id] == 1:
                c += ' [correct!]'
                # print(('\n\t\t[%.3f]'% score) + ' '.join(predict) + ' [correct!]')
                # print(('\n\t\t[%.3f]'% score) + ' '.join(predict))
        c += '\n'

        # c = '[DECODE]: {}'.format(' '.join(cut_zero(phrase, idx2word)))
        # if inputs_unk is not None:
        #     k = '[_INPUT]: {}\n'.format(' '.join(cut_zero(inputs_unk.tolist(),  idx2word, Lmax=len(idx2word))))
        #     logger.info(k)
        # a += k
        logger.info(c)
        a += b + c

        for number_to_predict in [5, 10, 15]:
            d = '@%d - Precision=%.4f, Recall=%.4f, F1=%.4f, Bpref=%.4f, MRR=%.4f' % (
            number_to_predict, metric_dict['p@%d' % number_to_predict], metric_dict['r@%d' % number_to_predict],
            metric_dict['f1@%d' % number_to_predict], metric_dict['bpref@%d' % number_to_predict], metric_dict['mrr@%d' % number_to_predict])
            logger.info(d)
            a += d + '\n'

        logger.info('*' * 100)
        outs.append(a)
        outs.append('*' * 100 + '\n')

    # omit the bad data which contains 0 predictions
    # real_test_size = sum([1 if m['target_number'] > 0 else 0 for m in micro_metrics])
    real_test_size = len(inputs)

    '''
    Compute the corpus evaluation
    '''
    overall_score = {}
    for k in [5, 10, 15]:
        correct_number = sum([m['correct_number@%d' % k] for m in micro_metrics])
        appear_target_number = sum([m['appear_target_number'] for m in micro_metrics])
        target_number = sum([m['target_number'] for m in micro_metrics])

        # Compute the Micro Measures, by averaging the micro-score of each prediction
        overall_score['p@%d' % k] = float(sum([m['p@%d' % k] for m in micro_metrics])) / float(real_test_size)
        overall_score['r@%d' % k] = float(sum([m['r@%d' % k] for m in micro_metrics])) / float(real_test_size)
        overall_score['f1@%d' % k] = float(sum([m['f1@%d' % k] for m in micro_metrics])) / float(real_test_size)

        output_str = 'Overall - %s valid testing data=%d, Number of Target=%d/%d, Number of Prediction=%d, Number of Correct=%d' % (
                    config['predict_type'], real_test_size,
                    appear_target_number, target_number,
                    real_test_size * k, correct_number
        )
        outs.append(output_str+'\n')
        logger.info(output_str)
        output_str = 'Micro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
                    k, overall_score['p@%d' % k],
                    k, overall_score['r@%d' % k],
                    k, overall_score['f1@%d' % k]
        )
        outs.append(output_str+'\n')
        logger.info(output_str)

        # Compute the Macro Measures
        overall_score['macro_p@%d' % k] = correct_number / float(real_test_size * k)
        overall_score['macro_r@%d' % k] = correct_number / float(appear_target_number)
        if overall_score['macro_p@%d' % k] + overall_score['macro_r@%d' % k] > 0:
            overall_score['macro_f1@%d' % k] = 2 * overall_score['macro_p@%d' % k] * overall_score[
                'macro_r@%d' % k] / float(overall_score['macro_p@%d' % k] + overall_score['macro_r@%d' % k])
        else:
            overall_score['macro_f1@%d' % k] = 0

        output_str = 'Macro:\t\tP@%d=%f, R@%d=%f, F1@%d=%f' % (
                    k, overall_score['macro_p@%d' % k],
                    k, overall_score['macro_r@%d' % k],
                    k, overall_score['macro_f1@%d' % k]
        )
        outs.append(output_str+'\n')
        logger.info(output_str)

        # Compute the binary preference measure (Bpref)
        overall_score['bpref@%d' % k] = float(sum([m['bpref@%d' % k] for m in micro_metrics])) / float(real_test_size)

        # Compute the mean reciprocal rank (MRR)
        overall_score['mrr@%d' % k] = float(sum([m['mrr@%d' % k] for m in micro_metrics])) / float(real_test_size)

        output_str = '\t\t\tBpref@%d=%f, MRR@%d=%f' % (
                    k, overall_score['bpref@%d' % k],
                    k, overall_score['mrr@%d' % k]
        )
        outs.append(output_str+'\n')
        logger.info(output_str)

    return outs, overall_score