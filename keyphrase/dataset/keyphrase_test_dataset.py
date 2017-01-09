#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python File Template 
"""

import os
import re
import shutil

import nltk
import xml.etree.ElementTree as ET

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

    def __str__(self):
        return '%s\n\t%s\n\t%s' % (self.title, self.text, str(self.phrases))


class DataLoader(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.basedir = self.basedir
        self.doclist = []

    def get_docs(self):
        '''
        :return: a list of dict instead of the Document object
        '''
        class_name = self.__class__.__name__.lower()
        if class_name == 'kdd' or class_name == 'www' or class_name == 'umd':
            self.load_xml(self.textdir)
        else:
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
        self.get_docs()

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

    def load_xml(self, xmldir):
        '''
        for KDD/WWW/UMD only
        :return: doclist
        '''
        for filename in os.listdir(xmldir):
            with open(xmldir+filename) as textfile:
                doc = Document()
                doc.name = filename[:filename.find('.xml')]

                import string
                printable = set(string.printable)

                # print((filename))
                try:
                    lines = textfile.readlines()
                    xml = ''.join([filter(lambda x: x in printable, l) for l in lines])
                    root = ET.fromstring(xml)

                    doc.title = root.findall("title")[0].text
                    doc.abstract = root.findall("abstract")[0].text
                    doc.phrases = [n.text for n in root.findall("*/tag")]

                    self.doclist.append(doc)

                except UnicodeDecodeError:
                    print('UnicodeDecodeError detected! %s' % filename )

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
            if os.path.exists(self.keyphrasedir + doc.name + '.keyphrases'):
                with open(keyphrasedir+doc.name+'.keyphrases') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
            # else:
            #     print(self.keyphrasedir + doc.name + '.keyphrases Not Found')

            if os.path.exists(self.keyphrasedir + doc.name + '.keywords'):
                with open(keyphrasedir + doc.name + '.keywords') as keyphrasefile:
                    phrase_set.update([phrase.strip() for phrase in keyphrasefile.readlines()])
            # else:
            #     print(self.keyphrasedir + doc.name + '.keywords Not Found')

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
            if os.path.exists(self.keyphrasedir + paper_id + '.keyphrases'):
                with open(self.keyphrasedir + paper_id + '.keyphrases', 'w') as f:
                    for key in list(keyphrases):
                        if key.find(' ') != -1:
                            f.write(key+'\n')
            # else:
            #     print(self.keyphrasedir + paper_id + '.keyphrases Not Found')

            if os.path.exists(self.keyphrasedir + paper_id + '.keywords'):
                with open(self.keyphrasedir + paper_id + '.keywords', 'w') as f:
                    for key in list(keyphrases):
                        if key.find(' ') == -1:
                            f.write(key+'\n')
            # else:
            #     print(self.keyphrasedir + paper_id + '.keywords Not Found')

class SemEval(DataLoader):
    def __init__(self, **kwargs):
        super(SemEval, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/SemEval'
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases_3/'

class KRAPIVIN(DataLoader):
    def __init__(self, **kwargs):
        super(KRAPIVIN, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/KRAPIVIN'
        self.textdir = self.datadir + '/all_texts/'
        self.keyphrasedir = self.datadir + '/gold_standard_keyphrases/'

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

                    title = lines[1].encode('ascii', 'ignore').decode('ascii', 'ignore')
                    # the 2nd line is abstract title
                    text  = lines[3].encode('ascii', 'ignore').decode('ascii', 'ignore')

                    # if lines[1].strip().lower() != 'abstract':
                    #     print('Wrong title detected : %s' % (filename))

                    doc.title = title
                    doc.text  = text
                    self.doclist.append(doc)

                except UnicodeDecodeError:
                    print('UnicodeDecodeError detected! %s' % filename )

class KDD(DataLoader):
    def __init__(self, **kwargs):
        super(KDD, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/KDD'
        self.textdir = self.datadir + '/acmparsed/'
        self.keyphrasedir = self.datadir + '/acmparsed/'

class WWW(DataLoader):
    def __init__(self, **kwargs):
        super(WWW, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/WWW'
        self.textdir = self.datadir + '/acmparsed/'
        self.keyphrasedir = self.datadir + '/acmparsed/'

class UMD(DataLoader):
    def __init__(self, **kwargs):
        super(UMD, self).__init__(**kwargs)
        self.datadir = self.basedir + '/dataset/keyphrase/testing-data/UMD'
        self.textdir = self.datadir + '/acmparsed/'
        self.keyphrasedir = self.datadir + '/acmparsed/'


# aliases
inspec = INSPEC
nus = NUS
semeval = SemEval
krapivin = KRAPIVIN
kdd = KDD
www = WWW
umd = UMD
# irbooks = IRBooks

def load_testing_data(identifier, kwargs=None):
    '''
    load testing data dynamically
    :return:
    '''
    data_loader = get_from_module(identifier, globals(), 'data_loader', instantiate=True,
                           kwargs=kwargs)
    return data_loader

if __name__ == '__main__':
    config = setup_keyphrase_all()
    train_set, validation_set, test_set, idx2word, word2idx = deserialize_from_file(config['dataset'])

    for dataset_name in config['testing_datasets']:
        print('*' * 50)
        print(dataset_name)
        loader = load_testing_data(dataset_name, kwargs=dict(basedir = config['path']))

        test_set, doclist = loader(idx2word, word2idx, type=0)
        test_data_plain = zip(*(test_set['source'], test_set['target'], doclist))

        for idx in xrange(len(test_data_plain)):  # len(test_data_plain)
            test_s_o, test_t_o, doc = test_data_plain[idx]
            target = doc.phrases


            if len(doc.text) < 50000:
                print('%d - %d : %d   \n\tname=%s, \n\ttitle=%s, \n\ttext=%s, \n\tlen(keyphrase)=%d' % (idx, len(test_s_o), len(test_t_o), doc.name, doc.title, doc.text, len(''.join(target))))
                print(doc)

            if (len(target)!=0 and len(''.join(target))/len(target) < 3):
                print('!' * 100)
                print('Error found')
                print('%d - %d : %d   name=%s, title=%d, text=%d, len(keyphrase)=%d' % (idx, len(test_s_o), len(test_t_o), doc.name, len(doc.title), len(doc.text), len(''.join(target))))

            # if len(test_t_o) < 3:
            #
            #     doc.text = re.sub(r'[\r\n\t]', ' ', doc.text)
            #     print('name:\t%s' % doc.name)
            #     print('text:\t%s' % doc.text)
            #     print('phrase:\t%s' % str(doc.phrases))
            # if idx % 100 == 0:
            #     print(test_data_plain[idx])