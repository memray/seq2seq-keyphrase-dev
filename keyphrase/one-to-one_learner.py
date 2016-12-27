import cPickle
import heapq
import os
from collections import defaultdict
from math import ceil
import sys
import time

import numpy as np
import re
from experimenter import ExperimentLogger

from convolutional_attention.copy_conv_rec_model import CopyConvolutionalRecurrentAttentionalModel
from convolutional_attention.f1_evaluator import F1Evaluator
from convolutional_attention.token_naming_data import TokenCodeNamingData


class ConvolutionalCopyAttentionalRecurrentLearner:

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.naming_data = None
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3
        self.parameters = None

    def train(self, input_file, model_path, patience=5, max_epochs=1000, minibatch_size=500):
        assert self.parameters is None, "Model is already trained"
        print "Extracting data..."
        # Get data (train, validation), split with ratio=0.92
        train_data, validation_data, self.naming_data = TokenCodeNamingData.get_data_in_recurrent_copy_convolution_format_with_validation(input_file, .92, self.padding_size)
        # name_targets: id array of groundtruth output
        # code_sentences: id array of input string after padding
        # code: original input string, length=N. each is the input doc, a list of the tokenized words
        # target_is_unk: if this term in groundtruth is unknown(out of vocabulary), size is equal to name_targets
        # copy_vectors: for each word in groundtruth, whether it can be copied from input(the position it appears in input). Size=N*target_word_count*sentence_word_count
        train_name_targets, train_code_sentences, train_code, train_target_is_unk, train_copy_vectors = train_data
        val_name_targets, val_code_sentences, val_code, val_target_is_unk, val_copy_vectors = validation_data

        # Create theano model and train
        if os.path.exists(model_file+'_tmp'):
            model = ConvolutionalCopyAttentionalRecurrentLearner.load(model_file+'_tmp')
        else:
            model = CopyConvolutionalRecurrentAttentionalModel(self.hyperparameters, len(self.naming_data.all_tokens_dictionary),
                                   self.naming_data.name_empirical_dist)

        def compute_validation_score_names():
            return model.log_prob_with_targets(val_code_sentences, val_copy_vectors, val_target_is_unk, val_name_targets)

        best_params = [p.get_value() for p in model.train_parameters]
        best_name_score = float('-inf')

        # record the rate of change
        ratios = np.zeros(len(model.train_parameters))
        n_batches = 0
        epochs_not_improved = 0

        # ----------------------------------------------------------------- #
        print "[%s] Starting training..." % time.asctime()
        for i in xrange(max_epochs):
            start_time = time.time()
            # 1. Clump minibatches:
            #       Shuffle the ordering, split into minibatches
            # generate an array [0, 1, 2 ... len(train_name_targets)-1]
            name_ordering = np.arange(len(train_name_targets), dtype=np.int32)
            np.random.shuffle(name_ordering)

            sys.stdout.write(str(i))
            # Clump minibatches, max minibatch size is 25?
            num_minibatches = min(int(ceil(float(len(train_name_targets)) / minibatch_size))-1, 25)

            # 2. Training.
            #   Feed in the data batch by batch
            for j in xrange(num_minibatches):
                name_batch_ids = name_ordering[j * minibatch_size:(j + 1) * minibatch_size]
                batch_code_sentences = train_code_sentences[name_batch_ids]
                for k in xrange(len(name_batch_ids)):
                    pos = name_batch_ids[k]
                    # input one data sample and compute the accumulate gradient
                    model.grad_accumulate(batch_code_sentences[k], train_copy_vectors[pos],
                                          train_target_is_unk[pos], train_name_targets[pos])
                assert len(name_batch_ids) > 0
                # Optimize by nesterov_rmsprop, return the change ratio
                ratios += model.grad_step()
                n_batches += 1
            sys.stdout.write("|")
            if i % 1 == 0:
                # get validation score
                name_ll = compute_validation_score_names()
                if name_ll > best_name_score:
                    best_name_score = name_ll
                    best_params = [p.get_value() for p in model.train_parameters]
                    print "At %s validation: name_ll=%s [best so far]" % (i, name_ll)
                    epochs_not_improved = 0
                else:
                    print "At %s validation: name_ll=%s" % (i, name_ll)
                    epochs_not_improved += 1
                # print the rate of changes
                for k in xrange(len(model.train_parameters)):
                    print "%s: %.0e" % (model.train_parameters[k].name, ratios[k] / n_batches)

                # reset ratios
                n_batches = 0
                ratios = np.zeros(len(model.train_parameters))
                # save the model
                model.save(model_path+'_tmp')

            # Early stop
            if epochs_not_improved >= patience:
                print "Not improved for %s epochs. Stopping..." % patience
                break
            elapsed = int(time.time() - start_time)
            print "Epoch elapsed %sh%sm%ss" % ((elapsed / 60 / 60) % 60, (elapsed / 60) % 60, elapsed % 60)
        print "[%s] Training Finished..." % time.asctime()

        # set the parameters to itself
        self.parameters = best_params
        model.restore_parameters(best_params)
        self.model = model

    identifier_matcher = re.compile('[a-zA-Z0-9]+')

    def get_copy_distribution(self, copy_weights, code):
        """
        Return a distribution over the copied tokens. Some tokens may be invalid (ie. non alphanumeric), there are
         excluded, but the distribution is not re-normalized. This is probabilistically weird, but it possibly lets
         the non-copy mechanism to recover.
         :param copy_weights:
         :param code:
        """
        token_probs = defaultdict(lambda: float('-inf')) # log prob of each token
        for code_token, weight in zip(code, copy_weights):
            if self.identifier_matcher.match(code_token) is not None:
                token_probs[code_token] = np.logaddexp(token_probs[code_token], np.log(weight))
        return token_probs

    def get_suggestions_for_next_subtoken(self, current_code, current_code_sentence, predicted_target_tokens_so_far):
        '''
        Given the input sentense and predicted_target_tokens_so_far,
        get the probability of predicted next word and return suggestions
        :param current_code: text of input
        :param current_code_sentence: token id of input after padding
        :param predicted_target_tokens_so_far:  predicted tokens so far
        :return:
            copy_prob: do copy or not
            suggestions: suggested words in descending order
            subtoken_target_logprob: predicted probabilities of next words
        '''
        # get the testing values with regards to the testing data (predicted_target_tokens_so_far)
        copy_weights, copy_prob, name_logprobs = self.model.copy_probs(predicted_target_tokens_so_far, current_code_sentence)
        # Get values for the last prediction (for the whole predicted_target_tokens_so_far)
        copy_weights, copy_prob, name_logprobs = copy_weights[-1], copy_prob[-1], name_logprobs[-1]
        # convert to probabilities
        copy_weights /= np.sum(copy_weights)
        # get the distribution after copy attention
        copy_dist = self.get_copy_distribution(copy_weights, current_code)

        # log prob of each subtoken by normal convolutional attention
        #       = np.log(1.-copy_prob) + name_logprobs[j]
        subtoken_target_logprob = defaultdict(lambda: float('-inf'))
        for j in xrange(len(self.naming_data.all_tokens_dictionary) - 1):
            subtoken_target_logprob[self.naming_data.all_tokens_dictionary.get_name_for_id(j)] = np.log(1. - copy_prob) + name_logprobs[j]

        # plus the log prob of copy attention
        copy_logprob = np.log(copy_prob)
        for word, word_copied_log_prob in copy_dist.iteritems():
            subtoken_target_logprob[word] = np.logaddexp(subtoken_target_logprob[word], copy_logprob + word_copied_log_prob)

        # get the next word suggestion, sorted by probability in descending order
        suggestions = sorted(subtoken_target_logprob.keys(), key=lambda x: subtoken_target_logprob[x], reverse=True)
        return copy_prob, suggestions, subtoken_target_logprob

    def predict_name(self, code, max_predicted_identifier_size=5, max_steps=100):
        '''
        Do BeamSearch to get the top few predictions
        :param code: input document string
        :param max_predicted_identifier_size: max length of output to predict. original=7, change to 5
        :param max_steps: max step of search
        :return:
        '''
        assert self.parameters is not None, "Model is not trained"
        code = code[0]

        # Preprocess to the input
        #    convert input into token_id
        code_sentence = [self.naming_data.all_tokens_dictionary.get_id_or_unk(tok) for tok in code]
        #    add padding
        padding = [self.naming_data.all_tokens_dictionary.get_id_or_unk(self.naming_data.NONE)]
        if self.padding_size % 2 == 0:
            code_sentence = padding * (self.padding_size / 2) + code_sentence + padding * (self.padding_size / 2)
        else:
            code_sentence = padding * (self.padding_size / 2 + 1) + code_sentence + padding * (self.padding_size / 2)

        code_sentence = np.array(code_sentence, dtype=np.int32)

        # Result: a list of tuple of full suggestions (token, prob)
        suggestions = defaultdict(lambda: float('-inf'))
        # A stack of candidates(partial suggestion )in the form ([subword1, subword2, ...], logprob)
        #   all the prediction starts with the SUBTOKEN_START <s>
        possible_suggestions_stack = [
            ([self.naming_data.SUBTOKEN_START], [], 0)]
        # Keep the max_size_to_keep (=15 here) suggestion scores (sorted in the heap).
        # Prune further exploration if something has already lower score
        predictions_probs_heap = [float('-inf')]
        max_size_to_keep = 15
        nsteps = 0
        '''
        Maintain a max(=15)heap of complete predictions have the highest probability so far.
        Do BeamSearch
            Extend the partial predictions
            At each step,
                -  Pick the highest probability prediction and predict its next subtokens, pushing them back to the heap.
                -  If when the </s> subtoken is generated, move the suggestion onto the list of suggestions
                -  Keep top k suggestions, prune partial suggestions that probability less than the current best kth full suggestion.
            Stop if search exceeds the max_steps(=100) or length of output exceeds max_predicted_identifier_size(=7)
            Limit the partial suggestion stack size=15
        '''
        while True:
            scored_list = []
            # Extend one more word for all the partial suggestions in stack
            while len(possible_suggestions_stack) > 0:
                # pop up a candidate to expand, a tuple of (list<token>, list<token>, prob)
                subword_tokens = possible_suggestions_stack.pop()

                # If we're done(</s> subtoken is generated), append to predictions_probs_heap (full suggestions)
                if subword_tokens[0][-1] == self.naming_data.SUBTOKEN_END:
                    final_prediction = tuple(subword_tokens[1][:-1])
                    # a failure prediction <s></s>
                    if len(final_prediction) == 0:
                        continue
                    log_prob_of_suggestion = np.logaddexp(suggestions[final_prediction], subword_tokens[2])
                    # Push only if the score is better than the current minimum and > 0 and remove extraneous entries
                    if log_prob_of_suggestion > predictions_probs_heap[0] and not log_prob_of_suggestion == float('-inf'):
                        suggestions[final_prediction] = log_prob_of_suggestion
                        heapq.heappush(predictions_probs_heap, log_prob_of_suggestion)
                        # keep the size of heap < max_size_to_keep(=15)
                        if len(predictions_probs_heap) > max_size_to_keep:
                            heapq.heappop(predictions_probs_heap)
                    continue
                # Stop recursion here if the size_of_prediction > max_predicted_identifier_size(=7)
                elif len(subword_tokens[1]) > max_predicted_identifier_size:
                    continue

                # Convert previous prediction (subword context) into id
                previous_subtokens = [self.naming_data.all_tokens_dictionary.get_id_or_unk(k) for k in subword_tokens[0]]
                previous_subtokens = np.array(previous_subtokens, dtype=np.int32)

                # Predict next subwords
                #       by calling the function get_suggestions_for_next_subtoken
                #       given input sentence and previous predictions
                copy_prob, next_subtoken_suggestions, subtoken_target_logprob \
                    = self.get_suggestions_for_next_subtoken(code, code_sentence, previous_subtokens)
                # Get the prob of unk
                subtoken_target_logprob["***"] = subtoken_target_logprob[self.naming_data.all_tokens_dictionary.get_unk()]

                def get_possible_options(subword_name):
                    '''
                    append the new token to the end of predicted tokens
                    return as the form of tuple of (list<token>, list<token>, prob)
                    first token list starts with <s>, second starts with word
                    '''
                    # TODO: Handle UNK differently?
                    # replace the predicted unk to ***
                    if subword_name == self.naming_data.all_tokens_dictionary.get_unk():
                        subword_name = "***"
                    name = subword_tokens[1] + [subword_name]
                    # TODO: Is this sound? plus the probability of this token to the whole prediction, longer the better?
                    return subword_tokens[0] + [subword_name], name, subtoken_target_logprob[subword_name] + \
                           subword_tokens[2]
                # Analyze the top 15 next tokens (higher numbers must be worse)
                possible_options = [get_possible_options(next_subtoken_suggestions[i]) for i in xrange(max_size_to_keep)]
                # Copy predictions from possible_options to scored_list
                #       Disallow suggestions that contain duplicated subtokens (last two words are same)
                scored_list.extend(filter(lambda x: len(x[1])==1 or x[1][-1] != x[1][-2], possible_options))

            # Prune, keep the ones have probabilities larger than the last one in heap
            scored_list = filter(lambda suggestion: suggestion[2] >= predictions_probs_heap[0] and suggestion[2] >= float('-inf'), scored_list)
            scored_list.sort(key=lambda entry: entry[2], reverse=True)

            # Update, set the top results to be the candidates to extend next round
            possible_suggestions_stack = scored_list[:max_size_to_keep]
            nsteps += 1
            if nsteps >= max_steps:
                break

        # Sort and append to predictions
        suggestions = [(identifier, np.exp(logprob)) for identifier, logprob in suggestions.items()]
        suggestions.sort(key=lambda entry: entry[1], reverse=True)
        # return suggestions
        return suggestions

    def save(self, filename):
        model_tmp = self.model
        del self.model
        with open(filename, 'wb') as f:
            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
        self.model = model_tmp

    @staticmethod
    def load(filename):
        """
        load model from model_file
        notice that the parameters are stored in learner.parameters
        you should call restore_parameters to load it into model (which is in model.train_parameters)
        :type filename: str
        :rtype: ConvolutionalAttentionalLearner
        """
        with open(filename, 'rb') as f:
            learner = cPickle.load(f)
        learner.model = CopyConvolutionalRecurrentAttentionalModel(learner.hyperparameters, len(learner.naming_data.all_tokens_dictionary),
                             learner.naming_data.name_empirical_dist)
        learner.model.restore_parameters(learner.parameters)
        return learner


def run_from_config(params, *args):
    if len(args) < 2:
        print "No input file or test file given: %s:%s" % (args, len(args))
        sys.exit(-1)
    input_file = args[0]
    test_file = args[1]
    if len(args) > 2:
        num_epochs = int(args[2])
    else:
        num_epochs = 1000

    params["D"] = 2 ** params["logD"]
    params["conv_layer1_nfilters"] = 2 ** params["log_conv_layer1_nfilters"]
    params["conv_layer2_nfilters"] = 2 ** params["log_conv_layer2_nfilters"]

    model = ConvolutionalCopyAttentionalRecurrentLearner(params)
    model.train(input_file, max_epochs=num_epochs)

    test_data, original_names = model.naming_data.get_data_in_recurrent_copy_convolution_format(test_file, model.padding_size)
    test_name_targets, test_code_sentences, test_code, test_target_is_unk, test_copy_vectors = test_data
    eval = F1Evaluator(model)
    point_suggestion_eval = eval.compute_names_f1(test_code, original_names,
                                                  model.naming_data.all_tokens_dictionary.get_all_names())
    return -point_suggestion_eval.get_f1_at_all_ranks()[1]


if __name__ == "__main__":
    sys.argv = 'copy_conv_rev_learner ../dataset/keyphrase/million-paper/processed_acm_title_abstract_keyword_one2one.json 100000 128 ../dataset/keyphrase/inspec/test.json'.split(' ')
    # sys.argv = 'copy_conv_rev_learner ../dataset/keyphrase/validation.json 1000 128 ../dataset/keyphrase/test.json'.split(' ')

    if len(sys.argv) < 5:
        print 'Usage <input_file> <max_num_epochs> d <test_file>'
        sys.exit(-1)

    input_file = sys.argv[1]
    max_num_epochs = int(sys.argv[2]) # batch size
    params = {
        "D": int(sys.argv[3]), # length of embedding
        "conv_layer1_nfilters": 32,
        "conv_layer2_nfilters": 16,
        "layer1_window_size": 18,
        "layer2_window_size": 19,
        "layer3_window_size": 2,
        "log_name_rep_init_scale": -1,
        "log_layer1_init_scale": -3.68,
        "log_layer2_init_scale": -4,
        "log_layer3_init_scale": -4,
        "log_hidden_init_scale": -1,
        "log_copy_init_scale":-0.5,
        "log_learning_rate": -3.05,
        "rmsprop_rho": .99,
        "momentum": 0.87,
        "dropout_rate": 0.4,
        "grad_clip":.75
    }

    params["train_file"] = input_file
    params["test_file"] = sys.argv[4]

    from logger.printlogger import PrintLogger
    sys.stdout = PrintLogger("ConvolutionalCopyAttentionalRecurrentLearner")

    with ExperimentLogger("ConvolutionalCopyAttentionalRecurrentLearner", params) as experiment_log:
        # Train a new model if no existing one
        model_file = "copy_convolutional_att_rec_model-" + os.path.basename(params["train_file"]) + ".pkl"
        print("Model path: " + model_file)
        if (not os.path.isfile(model_file)):
            print("Model doesn't exist, start training")
            model = ConvolutionalCopyAttentionalRecurrentLearner(params)
            model.train(input_file, model_file, max_epochs=max_num_epochs)
            model.save(model_file)
        # Load the model
        print("Model exists, loading model")
        model2 = ConvolutionalCopyAttentionalRecurrentLearner.load(model_file)

        # Load testing data and run test
        print("Loading testing data")
        test_data, original_names = model2.naming_data.get_data_in_recurrent_copy_convolution_format(params["test_file"], model2.padding_size)
        test_name_targets, test_code_sentences, test_code, test_target_is_unk, test_copy_vectors = test_data
        print("Testing")
        # Get the log_prob with regards to the given targets
        name_ll = model2.model.log_prob_with_targets(test_code_sentences, test_copy_vectors, test_target_is_unk, test_name_targets)
        print "Test name_ll=%s" % name_ll

        # Run F1 evaluation
        eval = F1Evaluator(model2)
        point_suggestion_eval = eval.compute_names_f1(test_code, original_names,
                                                      model2.naming_data.all_tokens_dictionary.get_all_names())
        print point_suggestion_eval
        results = point_suggestion_eval.get_f1_at_all_ranks()
        print results
        # record and print the evaluation results
        experiment_log.record_results({"f1_at_rank1": results[0], "f1_at_rank5":results[1]})
        print({"f1_at_rank1": results[0], "f1_at_rank5":results[1]})
