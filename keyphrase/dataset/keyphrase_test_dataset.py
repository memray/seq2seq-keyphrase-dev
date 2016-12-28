#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import re
import shutil

import nltk

from emolga.utils.generic_utils import get_from_module

import data_utils as utils
from keyphrase.config import setup_keyphrase_all, setup_keyphrase_all_testing
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
class Document(object):
    def __init__(self):
        self.name       = ''
        self.title      = ''
        self.text       = ''
        self.phrases    = []


class DataLoader(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.basedir = self.basedir
        self.doclist = []

    def get_docs(self):
        '''
        :return: a list of dict instead of the Document object
        '''
        self.load_text(self.textdir)
        self.load_keyphrase(self.keyphrasedir)

        doclist = []
        for d in self.doclist:
            newd = {}
            newd['abstract']    = re.sub('[\r\n]', ' ', d.text).strip()
            newd['title']       = re.sub('[\r\n]', ' ', d.title).strip()
            newd['keyword']     = ';'.join(d.phrases)
            doclist.append(newd)

        return doclist

    def __call__(self, idx2word, word2idx, type = 1):
        self.load_text(self.textdir)
        self.load_keyphrase(self.keyphrasedir)
        pairs = []

        for doc in self.doclist:
            try:
                title= utils.get_tokens(doc.title, type)
                text = utils.get_tokens(doc.text, type)
                if type == 0:
                    title.append('<eos>')
                elif type == 1:
                    title.append('.')

                title.extend(text)
                text = title

                # trunk, many texts are too long, would lead to out-of-memory
                if len(text) > 1500:
                    text = text[:1500]

                keyphrases = [utils.get_tokens(k, type) for k in doc.phrases]
                pairs.append((text, keyphrases))

            except UnicodeDecodeError:
                print('UnicodeDecodeError detected! %s' % doc.name)
            # print(text)
            # print(keyphrases)
            # print('*'*50)
        dataset = utils.build_data(pairs, idx2word, word2idx)

        return dataset, self.doclist

    def load_text(self, textdir):
        for filename in os.listdir(textdir):
            with open(textdir+filename) as textfile:
                doc = Document()
                doc.name = filename[:filename.find('.txt')]

                import string
                printable = set(string.printable)

                # print((filename))
                try:
                    lines = textfile.readlines()

                    lines = [filter(lambda x: x in printable, l) for l in lines]

                    title = lines[0].encode('ascii', 'ignore').decode('ascii', 'ignore')
                    # the 2nd line is abstract title
                    text  = (' '.join(lines[2:])).encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # if lines[1].strip().lower() != 'abstract':
                    #     print('Wrong title detected : %s' % (filename))

                    doc.title = title
                    doc.text  = text
                    self.doclist.append(doc)

                except UnicodeDecodeError:
                    print('UnicodeDecodeError detected! %s' % filename )

    def load_keyphrase(self, keyphrasedir):
        for doc in self.doclist:
            phrase_set = set()
            with open(keyphrasedir+doc.name+'.keyphrases') as keyphrasefile:
                phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])

            with open(keyphrasedir + doc.name + '.keywords') as keyphrasefile:
                phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])

            doc.phrases = list(phrase_set)

class INSPEC(DataLoader):
    def __init__(self, **kwargs):
        super(INSPEC, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/INSPEC'
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases_2/'

class NUS(DataLoader):
    def __init__(self, **kwargs):
        super(NUS, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/NUS'
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases/'

    def export(self):
        '''
        parse the original dataset into two folders: text and gold_standard_keyphrases
        :return:
        '''
        originaldir = self.datadir+'/original'
        for paper_id in os.listdir(originaldir):
            if os.path.isfile(originaldir+'/'+paper_id):
                continue
            # copy text to all_texts/
            text_file = originaldir+'/'+paper_id+'/'+paper_id+'.txt'
            shutil.copy2(text_file, self.textdir+'/'+paper_id+'.txt')
            # load keyphrases
            keyphrases = set()
            keyphrase_files = [originaldir+'/'+paper_id+'/'+paper_id+'.kwd']
            reader_phrase_dir = originaldir+'/'+paper_id+'/KEY/'
            for key_file in os.listdir(reader_phrase_dir):
                keyphrase_files.append(reader_phrase_dir+key_file)
            for key_file in keyphrase_files:
                with open(key_file, 'r') as f:
                    keyphrases.update(set([l.strip() for l in f.readlines()]))
            # write into gold_standard_keyphrases/
            with open(self.keyphrasedir + paper_id + '.keyphrases', 'w') as f:
                for key in list(keyphrases):
                    if key.find(' ') != -1:
                        f.write(key+'\n')
            with open(self.keyphrasedir + paper_id + '.keywords', 'w') as f:
                for key in list(keyphrases):
                    if key.find(' ') == -1:
                        f.write(key+'\n')

class SemEval(DataLoader):
    def __init__(self, **kwargs):
        super(SemEval, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/SemEval'
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases_3/'


# aliases
inspec = INSPEC
nus = NUS
semeval = SemEval
# irbooks = IRBooks

def load_testing_data(identifier, kwargs=None):
    '''
    load testing data dynamically
    :return:
    '''
    data_loader = get_from_module(identifier, globals(), 'optimizer', instantiate=True,
                           kwargs=kwargs)
    return data_loader

if __name__ == '__main__':
    config = setup_keyphrase_all_testing()
    train_set, test_set, idx2word, word2idx = deserialize_from_file(config['dataset'])

    for dataset_name in config['testing_datasets']:
        print('*' * 50)
        print(dataset_name)
        loader = load_testing_data(dataset_name, kwargs=dict(basedir = config['path']))

        test_set = loader(idx2word, word2idx, type=0)
        test_data_plain = zip(*(test_set['source'], test_set['target']))

        for idx in xrange(len(test_data_plain)):  # len(test_data_plain)
            test_s_o, test_t_o = test_data_plain[idx]
            doc = loader.doclist[idx]

            print('%d - %d : %d' % (idx, len(test_s_o), len(test_t_o)))
            # if len(test_t_o) < 3:
            #
            #     doc.text = re.sub(r'[\r\n\t]', ' ', doc.text)
            #     print('name:\t%s' % doc.name)
            #     print('text:\t%s' % doc.text)
            #     print('phrase:\t%s' % str(doc.phrases))
            # if idx % 100 == 0:
            #     print(test_data_plain[idx])