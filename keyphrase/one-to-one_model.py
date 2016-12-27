# coding=utf-8
import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from theanoutils.optimization import nesterov_rmsprop_multiple, logsumexp, dropout_multiple, log_softmax


class CopyConvolutionalRecurrentAttentionalModel(object):
    def __init__(self, hyperparameters, all_voc_size, empirical_name_dist):
        self.D = hyperparameters["D"]

        self.hyperparameters = hyperparameters
        self.__check_all_hyperparmeters_exist()
        self.all_voc_size = all_voc_size
        # empirical_name_dist is used for initiate name_bias only
        self.__init_parameter(empirical_name_dist)

    def __init_parameter(self, empirical_name_dist):
        # word-embedding
        all_name_rep = np.random.randn(self.all_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.all_name_reps = theano.shared(all_name_rep, name="code_name_reps")

        # What's name_bias here??????
        # By convention, the last one is NONE, which is never predicted.
        self.name_bias = theano.shared(np.log(empirical_name_dist)[:-1], name="name_bias")

        # ------------------------------- conv layer1 (filter)-------------------------------------------------------
        # init conv filters in layer1, number=self.hyperparameters["conv_layer1_nfilters"], size=self.hyperparameters["layer1_window_size"]* self.D
        conv_layer1_code = np.random.randn(self.hyperparameters["conv_layer1_nfilters"], 1,
                                     self.hyperparameters["layer1_window_size"], self.D) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_code = theano.shared(conv_layer1_code, name="conv_layer1_code")
        # init conv_bias in layer1, size = #(filter)
        conv_layer1_bias = np.random.randn(self.hyperparameters["conv_layer1_nfilters"]) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_bias = theano.shared(conv_layer1_bias, name="conv_layer1_bias")


        # ------------------------------- conv layer2 (filter)-------------------------------------------------------
        # Currently conflate all to one dimension
        # init conv layer2
        conv_layer2_code = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer1_nfilters"],
                                     self.hyperparameters["layer2_window_size"], 1) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_code = theano.shared(conv_layer2_code, name="conv_layer2_code")
        conv_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_bias = theano.shared(conv_layer2_bias, name="conv_layer2_bias")

        # ------------------------------- conv layer3, consists of conv_attr(alpha),copy_attr(kappa) and copy_coefficient(lambda)-----------------------------------------
        # copy_attr (kappa)
        #   Probability that each token will be copied
        conv_layer3_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_copy_code = theano.shared(conv_layer3_code, name="conv_layer3_copy_code")
        conv_layer3_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_copy_bias = theano.shared(conv_layer3_bias[0], name="conv_layer3_copy_bias")

        # copy_coefficient(lambda)
        #   Probability that we do a copy
        conv_copy_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_copy_init_scale"]
        self.conv_copy_code = theano.shared(conv_copy_code, name="conv_copy_code")
        conv_copy_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_copy_init_scale"]
        self.conv_copy_bias = theano.shared(conv_copy_bias[0], name="conv_copy_bias")

        # conv_attr(alpha)
        #   Attention vectors
        conv_layer3_att_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_att_code = theano.shared(conv_layer3_att_code, name="conv_layer3_att_code")
        conv_layer3_att_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_att_bias = theano.shared(conv_layer3_att_bias[0], name="conv_layer3_att_bias")

        # ------------------------------- Recurrent layer (seq2seq)-----------------------------------------
        gru_prev_hidden_to_next = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_next = theano.shared(gru_prev_hidden_to_next, name="gru_prev_hidden_to_next")
        gru_prev_hidden_to_reset = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_reset = theano.shared(gru_prev_hidden_to_reset, name="gru_prev_hidden_to_reset")
        gru_prev_hidden_to_update = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_update = theano.shared(gru_prev_hidden_to_update, name="gru_prev_hidden_to_update")

        gru_prediction_to_reset = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_reset = theano.shared(gru_prediction_to_reset, name="gru_prediction_to_reset")

        gru_prediction_to_update = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_update = theano.shared(gru_prediction_to_update, name="gru_prediction_to_update")

        gru_prediction_to_hidden = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_hidden = theano.shared(gru_prediction_to_hidden, name="gru_prediction_to_hidden")
        # why layer2_bias again?
        conv_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_bias = theano.shared(conv_layer2_bias, name="conv_layer2_bias")

        gru_hidden_update_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_hidden_update_bias = theano.shared(gru_hidden_update_bias, name="gru_hidden_update_bias")
        gru_update_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_update_bias = theano.shared(gru_update_bias, name="gru_update_bias")
        gru_reset_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_reset_bias = theano.shared(gru_reset_bias, name="gru_reset_bias")
        # what's this h0? h0 is the initial hidden unit parameter for GRU, size=#(conv_layer2_nfilters)
        h0 = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.h0 = theano.shared(h0, name="h0")

        self.rng = RandomStreams()
        # why sum up all the windows size as padding size?
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3

        self.train_parameters = [self.all_name_reps,
                                 self.conv_layer1_code, self.conv_layer1_bias,
                                 self.conv_layer2_code, self.conv_layer2_bias,
                                 self.conv_layer3_copy_code, self.conv_layer3_copy_bias,
                                 self.conv_copy_code, self.conv_copy_bias,self.h0,
                                 self.gru_prediction_to_reset, self.gru_prediction_to_hidden, self.gru_prediction_to_update,
                                 self.gru_prev_hidden_to_reset, self.gru_prev_hidden_to_next, self.gru_prev_hidden_to_update,
                                 self.gru_hidden_update_bias, self.gru_update_bias, self.gru_reset_bias,
                                 self.conv_layer3_att_code, self.conv_layer3_att_bias, self.name_bias]
        # define functions
        self.__compile_model_functions()

    def __check_all_hyperparmeters_exist(self):
        all_params = ["D",
                      "log_name_rep_init_scale",
                      "conv_layer1_nfilters", "layer1_window_size", "log_layer1_init_scale",
                      "conv_layer2_nfilters", "layer2_window_size", "log_layer2_init_scale",
                      "layer3_window_size", "log_layer3_init_scale",
                      "log_copy_init_scale", "log_hidden_init_scale",
                      "log_learning_rate", "momentum", "rmsprop_rho", "dropout_rate", "grad_clip"]
        for param in all_params:
            assert param in self.hyperparameters, param

    def restore_parameters(self, values):
        '''
        Restore the train_parameters from given values
        :param values:
        :return:
        '''
        for value, param in zip(values, self.train_parameters):
            param.set_value(value)
        self.__compile_model_functions()

    def __get_model_likelihood_for_sentence(self, sentence, name_targets, do_dropout=False, dropout_rate=0.5):
        '''
        Generate the whole symbolic graph
            here only return the predicting probabilities of all the words, not the objective function
        :param sentence:
        :param name_targets:
        :param do_dropout:
        :param dropout_rate:
        :return: sentence, name_targets: remain the same
                 copy_weights, attention_weights: copy/normal attention weights
                 copy_probs: the ratio between copy/normal
                 name_log_probs: the predicting probability of next word
                 filtered_features

        '''
        code_embeddings = self.all_name_reps[sentence] # SentSize x D
        #
        # 1. Parameters preparation, if needed do dropout
        #
        if do_dropout:
            # Conv part
            code_embeddings,  conv_weights_code_l1, conv_weights_code_l2, conv_weights_code_copy_l3,\
                conv_weights_code_do_copy, conv_weights_code_att_l3, \
                 = dropout_multiple(dropout_rate, self.rng, code_embeddings, self.conv_layer1_code,
                                    self.conv_layer2_code, self.conv_layer3_copy_code, self.conv_copy_code,
                                    self.conv_layer3_att_code)

            # GRU part
            gru_prediction_to_reset, gru_prediction_to_hidden, gru_prediction_to_update, \
                gru_prev_hidden_to_reset, gru_prev_hidden_to_next, gru_prev_hidden_to_update = \
                dropout_multiple(dropout_rate, self.rng,
                             self.gru_prediction_to_reset, self.gru_prediction_to_hidden, self.gru_prediction_to_update,
                             self.gru_prev_hidden_to_reset, self.gru_prev_hidden_to_next, self.gru_prev_hidden_to_update)
        else:
            conv_weights_code_l1 = self.conv_layer1_code
            conv_weights_code_l2 = self.conv_layer2_code
            conv_weights_code_copy_l3 = self.conv_layer3_copy_code
            conv_weights_code_do_copy = self.conv_copy_code
            conv_weights_code_att_l3 = self.conv_layer3_att_code

            gru_prediction_to_reset, gru_prediction_to_hidden, gru_prediction_to_update,\
                gru_prev_hidden_to_reset, gru_prev_hidden_to_next, gru_prev_hidden_to_update = \
                    self.gru_prediction_to_reset, self.gru_prediction_to_hidden, self.gru_prediction_to_update, \
                             self.gru_prev_hidden_to_reset, self.gru_prev_hidden_to_next, self.gru_prev_hidden_to_update
        #
        # 2. Build the symbolic graph for convolving (layer 1&2)
        #
        # build the symbolic graph for convolving a mini-batch of a stack of 2D inputs with a set of 2D filters
        code_convolved_l1 = T.nnet.conv2d(code_embeddings.dimshuffle('x', 'x', 0, 1), conv_weights_code_l1, image_shape=(1, 1, None, self.D),
                                          filter_shape=self.conv_layer1_code.get_value().shape)
        l1_out = code_convolved_l1 + self.conv_layer1_bias.dimshuffle('x', 0, 'x', 'x')
        # why keep original value only when the l1_out>0
        l1_out = T.switch(l1_out>0, l1_out, 0.1 * l1_out)

        code_convolved_l2 = T.nnet.conv2d(l1_out, conv_weights_code_l2, image_shape=(1, self.hyperparameters["conv_layer1_nfilters"], None, 1),
                                          filter_shape=self.conv_layer2_code.get_value().shape)
        l2_out = code_convolved_l2 + self.conv_layer2_bias.dimshuffle('x', 0, 'x', 'x')

        #
        # 3. Build the symbolic graph for GRU part by theano.scan
        #
        def step(
                # sequence parameter, the target word to be predicted
                target_token_id,
                # output parameter, though there're 6, but only hidden_state is specified
                hidden_state,
                # the following 19 parameter are in non-sequence parameter
                 attention_features,
                 gru_prediction_to_reset, gru_prediction_to_hidden, gru_prediction_to_update,
                 gru_prev_hidden_to_reset, gru_prev_hidden_to_next, gru_prev_hidden_to_update,
                 gru_hidden_update_bias, gru_update_bias, gru_reset_bias,
                 conv_att_weights_code_l3, conv_att_layer3_bias,
                 conv_weights_code_copy_l3, conv_layer3_copy_bias,
                 conv_weights_code_do_copy, conv_copy_bias,
                 code_embeddings, all_name_reps, use_prev_stat):
            '''
            Implement one step in GRU process.
            :param attention_features:
                        The step input to GRU, the input after two rounds of Convolution, before dot product(attentioning) with the h_t-1
                            It's static, stored in the non-sequence as l2_out
                        The real attention_features mentioned in paper (L_feat) is the gated_l2 below
                            It can be regarded as a basic soft-attention already
                            The intuition behind attention_features is that
                            given the input c, it uses convolution to compute k2 features for each location.
                            By then using ht−1 as a multiplicative gating-like mechanism,
                            only the currently relevant features are kept in L2.
            :returns
                next_hidden
                predicted_embedding: the attentionized embedding after normal convolutional attention (alpha)
                copy_pos_probs: copy attention weights (kappa)
                copy_prob: probability of doing copy (lambda)
                code_toks_weights: attention weights (alpha)
                gated_l2: the real attention feature (L_feat)
            '''

            # Real attentioned feature (L_feat), merge the embedding from L2 and the hidden state of last GRU layer h_t-1
            #   Refer to the function <b>attention_features</b> in papaer
            # TODO What's the point here about hidden_state>0???
            gated_l2 = attention_features * T.switch(hidden_state>0, hidden_state, 0.01 * hidden_state).dimshuffle(0, 1, 'x', 'x')
            gated_l2 = gated_l2 / gated_l2.norm(2)

            # -------------  Normal Attention (alpha) -----------------
            # Get the normal attention weights based on a Convolution
            code_convolved_l3 = T.nnet.conv2d(gated_l2, conv_att_weights_code_l3,
                                              image_shape=(1, self.hyperparameters["conv_layer2_nfilters"], None, 1),
                                              filter_shape=self.conv_layer3_att_code.get_value().shape)[:, 0, :, 0]
            l3_out = code_convolved_l3 + conv_att_layer3_bias
            # attention weights (alpha), size = 1*LEN(c), size of the sentence
            code_toks_weights = T.nnet.softmax(l3_out)  # This should be one dimension (the size of the sentence)

            # The input embedding after normal conv attention
            #   shape(A=code_toks_weights) = 1*S, shape(Bcode_embeddings) = S*D
            #   so tensordot(A*B) = 1*D, i.e.the weighted embedding of whole sentence
            #   but no normalization? may be normalized by the latter softmax
            predicted_embedding = T.tensordot(code_toks_weights, code_embeddings[self.padding_size/2 + 1:-self.padding_size/2 + 1], [[1], [0]])[0]

            # ------------- Copy Attention (kappa) --------------------
            # Get the copy attention weights based on a Convolution
            code_copy_convolved_l3 = T.nnet.conv2d(gated_l2, conv_weights_code_copy_l3,
                                          image_shape=(1, self.hyperparameters["conv_layer2_nfilters"], None, 1),
                                          filter_shape=self.conv_layer3_copy_code.get_value().shape)[:, 0, :, 0]
            copy_l3_out = code_copy_convolved_l3 + conv_layer3_copy_bias
            # Get the copy attention weights (kappa), the size of the sentence
            copy_pos_probs = T.nnet.softmax(copy_l3_out)[0]  # This should be one dimension (the size of the sentence)

            # ------------- Meta Attention (lambda) --------------------
            # λ = max(σ(CONV1D(L_feat, K_λ)))
            #      L_feat is the attention feature, K_λ is a kernal
            # but why do max to get a static value?
            do_copy_code = T.max(T.nnet.conv2d(gated_l2, conv_weights_code_do_copy,
                                              image_shape=(1, self.hyperparameters["conv_layer2_nfilters"], None, 1),
                                              filter_shape=self.conv_copy_code.get_value().shape)[:, 0, :, 0])
            # Get probability of doing copy by sigmoid
            copy_prob = T.nnet.sigmoid(do_copy_code + conv_copy_bias)

            # --------------- Get the next hidden! Assuming the current prediction is correct-----------------
            # Do dropout
            if do_dropout:
                # For regularization, we can use the context embeddings *some* of the time
                embedding_used = T.switch(use_prev_stat, all_name_reps[target_token_id], predicted_embedding)
            else:
                # Or just use the target word as current input
                embedding_used = all_name_reps[target_token_id]

            reset_gate = T.nnet.sigmoid(
                T.dot(embedding_used, gru_prediction_to_reset) + T.dot(hidden_state, gru_prev_hidden_to_reset) + gru_reset_bias
            )
            update_gate = T.nnet.sigmoid(
                T.dot(embedding_used, gru_prediction_to_update) + T.dot(hidden_state, gru_prev_hidden_to_update) + gru_update_bias
            )
            hidden_update = T.tanh(
                T.dot(embedding_used, gru_prediction_to_hidden) + reset_gate * T.dot(hidden_state, gru_prev_hidden_to_next) + gru_hidden_update_bias
            )

            next_hidden = (1. - update_gate) * hidden_state + update_gate * hidden_update

            return next_hidden, predicted_embedding, copy_pos_probs, copy_prob, code_toks_weights, gated_l2


        # store the non-sequence parameters and feed into the theano.scan
        use_prev_stat = self.rng.binomial(n=1, p=1.-dropout_rate)
        non_sequences = [l2_out,
                         gru_prediction_to_reset, gru_prediction_to_hidden, gru_prediction_to_update, # GRU
                         gru_prev_hidden_to_reset, gru_prev_hidden_to_next, gru_prev_hidden_to_update,
                         self.gru_hidden_update_bias, self.gru_update_bias, self.gru_reset_bias,
                         conv_weights_code_att_l3, self.conv_layer3_att_bias,  # Normal Attention
                         conv_weights_code_copy_l3, self.conv_layer3_copy_bias, # Copy Attention
                         conv_weights_code_do_copy, self.conv_copy_bias, # Do we copy?
                         code_embeddings, self.all_name_reps, use_prev_stat]
        # scan return two parts, results returned by scan and the useless updates(_)
        # outputs_info(prior result), must be of a shape similar to that of the output variable generated at each iteration and moreover
        # Note that the result contains the value for each step.
        [h, predictions, copy_weights, copy_probs, attention_weights, filtered_features], _ = \
                                                                           theano.scan(step, sequences=name_targets,
                                                                           outputs_info=[self.h0, None, None, None, None, None],
                                                                           name="target_name_scan",
                                                                           non_sequences=non_sequences, strict=True)
        #
        # 4. Get the probabilities of all words on each prediction step
        # Get the predicting probabilities by Softmax(1*D x D*V + b)
        #       predictions: 1*D, the attention-weighted embedding of sentence
        #              I think here's a bug: the sentence embedding after normal attention only?
        #       self.all_name_reps: V*D, all words embedding
        #    Result: the dot results in a value for each word(similarity to the senvec),
        #               and log_softmax maps it into probibility
        #    Intuition: The more likely a word to be good candidates, the more like it's vector to the sentence vector
        #                 But does this really make sense?
        name_log_probs = log_softmax(T.dot(predictions, T.transpose(self.all_name_reps[:-1])) + self.name_bias) # SxD, DxK -> SxK
        return sentence, name_targets, copy_weights, attention_weights, copy_probs, name_log_probs, filtered_features

    def model_objective(self, copy_probs, copy_weights, is_copy_matrix, name_log_probs, name_targets, targets_is_unk):
        '''
        Return the probability of correct prediction words, our goal is to maximize it

        Refer to: http://cs231n.github.io/linear-classify/#softmax
            Minimizing the negative log likelihood of the correct class,
            which can be interpreted as performing Maximum Likelihood Estimation (MLE)
              - goal is to maximize the probability of correct class(next word)
              - thus minimize the cross-entropy loss, comparing the prediction with the correct label(word)
        :param copy_probs: lambda, the ratio of doing copy
        :param copy_weights: copy attention weights, size=len(sentence)
        :param is_copy_matrix: the word should copy (correct copy), binary array
        :param name_log_probs: probability of predicted next word by GRU
        :param name_targets: the word should predict (correct prediction)
        :param targets_is_unk: penalty for unk?
        :return:
        '''
        # 1. log(Copy part)
        # if there is at least one position to copy from(is_copy_matrix>0)
        #   then we do copy, copy_probs*copy_weights*copy_candidates: T.log(copy_probs) + T.log(T.sum(is_copy_matrix * copy_weights, axis=1)+10e-8)
        #   or return a big negative value: -1000   make sense?
        # TODO: Fix
        use_copy_prob = T.switch(T.sum(is_copy_matrix, axis=1) > 0, T.log(copy_probs) + T.log(T.sum(is_copy_matrix * copy_weights, axis=1)+10e-8), -1000)

        # 2. log(Generate part), three terms
        #       1. if target is unk, give a penalty=-10 (you should copy somehow)
        #       2. the non-copy coefficient = T.log(1. - copy_probs)
        #       3. the probabilities of correct words
        use_model_prob = T.switch(targets_is_unk, -10, 0) + T.log(1. - copy_probs) + name_log_probs[T.arange(name_targets.shape[0]), name_targets]

        # 3. sum up these two, notice that log(a+b)!=log(a)+log(b), so we use logsumexp trick to approximate
        correct_answer_log_prob = logsumexp(use_copy_prob, use_model_prob)

        # mean of what? seems to be of all samples in the batch
        return T.mean(correct_answer_log_prob)

    def __compile_model_functions(self):
        '''
        define all the functions to connect each module
        :return:
        '''
        # a variable stores the parameter updates, initialized by 0
        #   last one is the number of sentences have processed
        grad_acc = [theano.shared(np.zeros(param.get_value().shape)) for param in self.train_parameters] \
                    + [theano.shared(0, name="sentence_count")]

        sentence = T.ivector("sentence")
        is_copy_matrix = T.imatrix("is_copy_matrix")
        name_targets = T.ivector("name_targets")
        targets_is_unk = T.ivector("targets_is_unk")

        # theano.config.compute_test_value = 'warn'
        sentence.tag.test_value = np.arange(105).astype(np.int32)
        name_targets.tag.test_value = np.arange(5).astype(np.int32)
        targets_is_unk.tag.test_value = np.array([0, 0, 1, 0, 0], dtype=np.int32)
        is_copy_test_value = [[i % k == k-2 for i in xrange(105 - self.padding_size)] for k in [1, 7, 10, 25, 1]]
        is_copy_matrix.tag.test_value = np.array(is_copy_test_value, dtype=np.int32)

        # ------------------ Training part ---------------------#

        # Get the predicting probability of target words
        #   Generate the compiling graphs for training
        #   - return the predicting probabilities of all the words
        #   - input the sentence and name_targets (an array of token id)
        _, _, copy_weights, _, copy_probs, name_log_probs, _\
                = self.__get_model_likelihood_for_sentence(sentence, name_targets, do_dropout=True,
                                                       dropout_rate=self.hyperparameters["dropout_rate"])
        # Objective function, the predicting probability of correct answer
        correct_answer_log_prob = self.model_objective(copy_probs[:-1], copy_weights[:-1], is_copy_matrix[1:], name_log_probs[:-1], name_targets[1:], targets_is_unk[1:])

        # Return symbolic gradients for one or more variables with respect to some cost.
        #   Gradient of each parameter, the true grad is gonna be gotten in optimization
        grad = T.grad(correct_answer_log_prob, self.train_parameters)

        # Callable function for training, feeding in one data sample at a time, get accumulated gradients
        # Update grad_acc: update v=v+g (accumulated gradients so far) and sentence_count+=1
        self.grad_accumulate = theano.function(inputs=[sentence, is_copy_matrix, targets_is_unk, name_targets],
                                               updates=[(v, v+g) for v, g in zip(grad_acc, grad)] + [(grad_acc[-1],                                                        grad_acc[-1]+1)],#mode='NanGuardMode'
                                               )

        # normalize gradients by dividing by the count of samples
        normalized_grads = [T.switch(grad_acc[-1] > 0, g / grad_acc[-1], g) for g in grad_acc[:-1]]

        # optimize by the rmsprop
        step_updates, ratios = nesterov_rmsprop_multiple(self.train_parameters, normalized_grads,
                                                learning_rate=10 ** self.hyperparameters["log_learning_rate"],
                                                rho=self.hyperparameters["rmsprop_rho"],
                                                momentum=self.hyperparameters["momentum"],
                                                grad_clip=self.hyperparameters["grad_clip"],
                                                output_ratios=True)
        # Copy the parameters to the end of step_updates. grad_acc[:-1] stores the model parameters, grad_acc[-1] is the number of data
        step_updates.extend([(v, T.zeros(v.shape)) for v in grad_acc[:-1]])  # Set accumulators to 0
        # Set the accumulators to 0
        step_updates.append((grad_acc[-1], 0))

        # Callable function for optimizing by rmsprop, to get the ratios outputs
        #       updates - expressions indicating how to update
        #       outputs - expressions to compute.
        self.grad_step = theano.function(inputs=[], updates=step_updates, outputs=ratios)

        # ------------------ Testing part ---------------------#
        # Generate the compiling graphs for testing
        #       don't do dropout on test
        test_sentence, test_name_targets, test_copy_weights, test_attention_weights, test_copy_probs, test_name_log_probs,\
                         test_attention_features \
            = self.__get_model_likelihood_for_sentence(T.ivector("test_sentence"),  T.ivector("test_name_targets"),
                                                      do_dropout=False)

        # functions to get probabilities with regards to the testing data
        #       param: test_name_targets: targets
        #       param: test_sentence: sentence input
        #       return: test_copy_weights, test_copy_probs, test_name_log_probs)
        self.copy_probs = theano.function(inputs=[test_name_targets, test_sentence],
                                                  outputs=[test_copy_weights, test_copy_probs, test_name_log_probs])

        test_copy_matrix = T.imatrix("test_copy_matrix")
        test_target_is_unk = T.ivector("test_target_is_unk")

        # the log_prob with the given targets
        ll = self.model_objective(test_copy_probs[:-1], test_copy_weights[:-1], test_copy_matrix[1:], test_name_log_probs[:-1], test_name_targets[1:], test_target_is_unk[1:])
        # function to get probability of correct prediction words
        self.copy_logprob = theano.function(inputs=[test_sentence, test_copy_matrix, test_target_is_unk, test_name_targets], outputs=ll)
        # function to get test_attention_weights
        self.attention_weights = theano.function(inputs=[test_name_targets, test_sentence],
                                                 outputs=test_attention_weights)
        layer3_padding = self.hyperparameters["layer3_window_size"] - 1
        upper_pos = -layer3_padding/2+1 if -layer3_padding/2+1 < 0 else None
        self.attention_features = theano.function(inputs=[test_sentence, test_name_targets],
                                                  outputs=test_attention_features[-1, 0, :, layer3_padding/2+1:upper_pos, 0])

    def log_prob_no_predict(self, name_contexts, sentences, copy_vectors, target_is_unk, name_target):
        '''
        Not used
        '''
        ll = 0
        for i in xrange(len(sentences)):
            ll += self.copy_logprob(name_contexts[i], sentences[i], copy_vectors[i], target_is_unk[i], name_target[i])
        return (ll / len(sentences))

    def log_prob_with_targets(self, sentence, copy_matrices, targets_is_unk, name_targets):
        '''
        Return the log_prob with regards to the given targets
        :param sentence:
        :param copy_matrices:
        :param targets_is_unk:
        :param name_targets:
        :return:
        '''
        ll = 0
        for i in xrange(len(name_targets)):
            ll += self.copy_logprob(sentence[i], copy_matrices[i], targets_is_unk[i], name_targets[i])
        return (ll / len(name_targets))

