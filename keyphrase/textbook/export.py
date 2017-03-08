# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import numpy as np
from keyphrase import keyphrase_utils
from keyphrase.dataset.keyphrase_test_dataset import testing_data_loader, load_additional_testing_data
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
from keyphrase.config import *

config = setup_keyphrase_all # setup_keyphrase_all_testing

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    config = setup_keyphrase_all()  # load settings.

    # loader = testing_data_loader('irbooks', kwargs=dict(basedir=config['path']))
    # docs   = loader.get_docs(return_dict=True)

    train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    test_sets = load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config,
                                                                    postagging=True, process_type=2)
    #
    #  test_set, test_s_list, test_t_list, test_s_o_list, test_t_o_list, input_encodings, predictions, scores, output_encodings, idx2word \
    #     = deserialize_from_file(config['predict_path'] + 'predict.{0}.{1}.pkl'.format(config['predict_type'], 'irbooks'))
    #
    # do_stem = False
    #
    # # Evaluation
    # outs, overall_score = keyphrase_utils.evaluate_multiple(config, test_set, test_s_list, test_t_list,
    #                                                         test_s_o_list, test_t_o_list,
    #                                                         predictions, scores, idx2word, do_stem,
    #                                                         model_name=config['task_name'],
    #                                                         dataset_name='irbooks')
