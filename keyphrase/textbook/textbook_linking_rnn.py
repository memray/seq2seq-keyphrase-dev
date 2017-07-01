# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
from keyphrase.config import *
from sklearn.metrics.pairwise import cosine_similarity
from rank_metrics import ndcg_at_k
import numpy as np

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"


def load_evaluation_data(EVALUATION_DATA_PATH):
    '''
    Load the ground truth and generate the training/testing data by shuffling
    :param GROUNDTRUTH_PATH:
    :return:
    '''
    testing_data_dict = []
    training_data_dict =[]

    file_num = len(os.listdir(EVALUATION_DATA_PATH))
    for i in range(file_num/2):
        testing_data_dict.append({})
        training_data_dict.append({})

    for file_name in os.listdir(EVALUATION_DATA_PATH):
        if file_name.startswith('train'):
            file_number = int(file_name[file_name.index('_')+1:-4])
            file = open(EVALUATION_DATA_PATH+file_name,'r')
            for line in file:
                doc_id = line[0: line.index('\t')].strip()
                mappings = line[line.index('\t')+1:].strip().split(',')
                mapping_dict = {}
                for mapping in mappings:
                    mapping_dict[mapping.split(':')[0]]=float(mapping.split(':')[1])
                training_data_dict[file_number][doc_id] = mapping_dict

    for file_name in os.listdir(EVALUATION_DATA_PATH):
        if file_name.startswith('test'):
            file_number = int(file_name[file_name.index('_')+1:-4])
            file = open(EVALUATION_DATA_PATH+file_name,'r')
            for line in file:
                doc_id = line[0: line.index('\t')].strip()
                mappings = line[line.index('\t')+1:].strip().split(',')
                mapping_dict = {}
                for mapping in mappings:
                    mapping_dict[mapping.split(':')[0]]=float(mapping.split(':')[1])
                testing_data_dict[file_number][doc_id] = mapping_dict

    return training_data_dict, testing_data_dict


def load_mir_names():
    with open(config['path'] + '/dataset/keyphrase/testing-data/IRbooks/mir_textbook_old.txt', 'r') as f:
        lines = f.readlines()
        name_map = {}
        for l in lines:
            old_name = l.split('\t')[0].strip()
            new_name = 'mir_'+l.split('\t')[1].strip().replace('.','_')
            name_map[new_name] = old_name
    return name_map


def evaluate_ndcg_at_k(testing_data, k=3):
    '''
    calc the similarity based on merging term and topic model
    :param: k, top k results accounted for calculating NDCG
    :param: lambda_ratio, the ratio for blending, final_similarity = lambda*term_similarity + (1-lambda)*topic_model_similarity
            lambda_ratio = 0 means only topic_model_similarity.
            lambda_ratio = 1 means only term_similarity
    :return: return NDCG@k
    '''
    ndcg_total = 0
    query_number = 0

    '''
    Get the final similarity rank
    '''
    # print('Getting the final similarity rank')
    for iir_name, mapping_dict in testing_data.items():
        if not iir_name in similarity_matrix:
            # print ('%s not found' % iir_name)
            continue

        weighted_similarities = similarity_matrix[iir_name]
        weighted_similarities = sorted(weighted_similarities, key=lambda item: item[1],reverse=True)

        r_array = [] # array used as input of nDCG
        for entry in weighted_similarities:
            if entry[0] in mapping_dict:
                r_array.append(mapping_dict[entry[0]])
            else:
                r_array.append(0)
                # print(entry[0], entry[0], entry[1])
        ndcg = ndcg_at_k(r_array, k)
        # print(r_array)
        # print(ndcg)
        ndcg_total += ndcg
    return float(ndcg_total)/len(testing_data)


if __name__ == '__main__':
    config = setup_keyphrase_all()  # load settings.

    training_data_dict, testing_data_list = load_evaluation_data(config['path'] + '/dataset/textbook_linking/ir/')

    docs = deserialize_from_file(config['path'] + '/dataset/textbook_linking/docs.pkl')

    iir_docs = [d for d in docs if d['name'].startswith('iir')]
    mir_docs = [d for d in docs if d['name'].startswith('mir') and d['name']!='mir_10_5_2']

    mir_name_map = load_mir_names()

    for d in mir_docs:
        # print('%s -> %s' % (d['name'], mir_name_map[d['name']]))
        d['name']=mir_name_map[d['name']]

    similarity_matrix = {}
    encoding_name = 'backward'
    similarity_matrix_file = config['path']+'/dataset/textbook_linking/similarity_matrix_'+encoding_name+'.pkl'
    if os.path.exists(similarity_matrix_file):
        similarity_matrix = deserialize_from_file(similarity_matrix_file)
    else:
        for iir_doc in iir_docs:
            similarity_matrix[iir_doc['name']] = []
            for mir_doc in mir_docs:
                if encoding_name == 'forward':
                    iir_vec = iir_doc['forward_encoding']
                    mir_vec = mir_doc['forward_encoding']

                if encoding_name == 'backward':
                    iir_vec = iir_doc['backward_encoding']
                    mir_vec = mir_doc['backward_encoding']

                if encoding_name == 'forward-backward':
                    iir_vec = np.concatenate([iir_doc['forward_encoding'], iir_doc['backward_encoding']])
                    mir_vec = np.concatenate([mir_doc['forward_encoding'], mir_doc['backward_encoding']])

                sim = cosine_similarity(iir_vec, mir_vec)
                similarity_matrix[iir_doc['name']].append((mir_doc['name'], sim))
                # print('%s vs %s = %f' % (iir_doc['name'], mir_doc['name'], sim))

        serialize_to_file(similarity_matrix, similarity_matrix_file)

    for k in [1,3,5]:
        ndcg_k = 0
        for testing_data_i in testing_data_list:
            '''
            each testing_data_i consists of a bunch of mappings
            '''
            # print(len(testing_data_i))
            ndcg_ = evaluate_ndcg_at_k(testing_data_i, k)
            ndcg_k += ndcg_
        print('NDCG@%d = %f/%d = %f' % (k, ndcg_k, len(testing_data_list), float(ndcg_k)/len(testing_data_list)))