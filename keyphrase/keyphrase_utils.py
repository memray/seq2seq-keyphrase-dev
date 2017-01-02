import math

import logging
from nltk.stem.porter import *
import numpy as np

logger = logging.getLogger(__name__)

def evaluate_multiple(config, inputs, outputs,
                      original_input, original_outputs,
                      samples, scores, idx2word):
    '''
    inputs_unk is same as inputs except for filtered out all the low-freq words to 1 (<unk>)
    return the top few keywords, number is set in config
    :param: original_input, same as inputs, the vector of one input sentence
    :param: original_outputs, vectors of corresponding multiple outputs (e.g. keyphrases)
    :return:
    '''

    def cut_zero(sample, idx2word, Lmax=None):
        sample = list(sample)
        if Lmax is None:
            Lmax = config['dec_voc_size']
        if 0 not in sample:
            return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample]
        # return the string before 0 (<eol>)
        return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample[:sample.index(0)]]

    # Generate keyphrases
    # if inputs_unk is None:
    #     samples, scores = self.generate_multiple(inputs[None, :], return_all=True)
    # else:
    #     samples, scores = self.generate_multiple(inputs_unk[None, :], return_all=True)

    stemmer = PorterStemmer()
    # Evaluation part
    outs = []
    metrics = []

    # load stopword
    with open(config['path'] + '/dataset/stopword/stopword_en.txt') as stopword_file:
        stopword_set = set([stemmer.stem(w.strip()) for w in stopword_file])

    for input_sentence, target_list, predict_list, score_list in zip(inputs, original_outputs, samples, scores):
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
            target = cut_zero(target, idx2word)
            target = [stemmer.stem(w) for w in target]

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
                    print('\t\tPunctuations! - %s' % str(predict))
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
                print('\t\tall letters! - %s' % str(predict))

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

        for number_to_predict in [5, 10, 15]:
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

            metric_dict['valid_target_number'] = len(target_outputs)
            metric_dict['target_number'] = len(target_list)
            metric_dict['correct_number@%d' % number_to_predict] = sum(correctly_matched[:number_to_predict])

        metrics.append(metric_dict)

        # print stuff
        a = '[SOURCE]: {}'.format(' '.join(cut_zero(input_sentence, idx2word)))
        logger.info(a)

        b = '[TARGET]: %d/%d targets\n\t\t' % (len(target_outputs), len(target_list))
        for id, target in enumerate(target_outputs):
            b += ' '.join(target) + '; '
        logger.info(b)
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
            d = '@%d - Precision=%.4f, Recall=%.4f, F1=%.4f' % (
            number_to_predict, metric_dict['p@%d' % number_to_predict], metric_dict['r@%d' % number_to_predict],
            metric_dict['f1@%d' % number_to_predict])
            logger.info(d)
            a += d

        outs.append(a)

    return outs, metrics