#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean the dirty format of keyword files
Convert to one keyword per line
"""
import json
import os
import re
from keyphrase.config import setup_keyphrase_all

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    basepath = os.path.realpath(os.path.curdir)

    INPUT_FILE      = basepath+"/dataset/keyphrase/ir-books/expert-conflict-free/expert-conflict-free-no-stemming.txt"
    TEXT_DIR        = basepath+"/dataset/keyphrase/ir-books/ir_documents/"
    OUTPUT_FILE     = basepath+"/dataset/keyphrase/ir-books/expert-conflict-free.json"
    # os.makedirs(OUTPUT_DIR)

    output_list = []

    with open(INPUT_FILE) as input_file:
        for line in input_file:
            dict = {}
            doc_name = line[:line.find('\t')]
            keywords = line.strip()[line.find('\t')+1:-1].split(';')
            # print(doc_name)
            # print(keywords)

            print(TEXT_DIR+doc_name+'.txt')
            doc_file = open(TEXT_DIR+doc_name+'.txt', 'r')
            text = ' '.join([line.strip() for line in doc_file.readlines()])

            dict['abstract'] = text
            dict['title']    = doc_name
            dict['keyword']  = ';'.join(keywords)

            output_list.append(dict)

    with open(OUTPUT_FILE, 'w') as output_file:
        output_file.write(json.dumps(output_list))