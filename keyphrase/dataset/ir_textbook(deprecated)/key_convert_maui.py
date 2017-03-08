#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean the dirty format of keyword files
Convert to one keyword per line
"""

import os
import re

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    INPUT_FILE = "/home/memray/Project/keyphrase/maui/src/test/resources/data/ir_books/expert-non-agree/expert-conflict-free-no-stemming.txt"
    TEXT_DIR = "/home/memray/Project/keyphrase/maui/src/test/resources/data/ir_books/ir_documents/"
    KEY_DIR = "/home/memray/Project/keyphrase/maui/src/test/resources/data/ir_books/expert-non-agree/key/"
    OUTPUT_DIR = "/home/memray/Project/keyphrase/maui/src/test/resources/data/ir_books/expert-non-agree/test/"
    # os.makedirs(OUTPUT_DIR)

    with open(INPUT_FILE) as input_file:
        # extract keyphrase and write into key/ folder
        for line in input_file:
            doc_name = line[:line.find('\t')]
            keywords = line.strip()[line.find('\t')+1:-1].split(';')
            # print(doc_name)
            # print(keywords)

            with open(KEY_DIR+doc_name+ '.key', 'w') as output_file:
                for keyword in keywords:
                    keyword = re.sub('\s+',' ',keyword)
                    print(keyword+'\t1\n')
                    output_file.write(keyword+'\t1\n')

        # copy .key and .txt into the test/ folder
        import subprocess
        bashCommand = "cp "+TEXT_DIR+"* "+OUTPUT_DIR
        print(bashCommand)
        process = subprocess.call(bashCommand, shell=True)
        bashCommand = "cp "+KEY_DIR+"* "+OUTPUT_DIR
        print(bashCommand)
        process = subprocess.call(bashCommand, shell=True)
        bashCommand = "rename 's/\.html$/\.txt/' "+OUTPUT_DIR+"*.html"
        print(bashCommand)
        process = subprocess.call(bashCommand, shell=True)

