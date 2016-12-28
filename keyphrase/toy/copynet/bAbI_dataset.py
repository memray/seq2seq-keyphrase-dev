"""
Preprocess the bAbI datset.
"""
import logging
import os
import sys
import numpy.random as n_rng
from emolga.dataset.build_dataset import serialize_to_file
from config import *

config  = setup_bAbI()
data_path = config['input_dataset']
data = []
n_rng.seed(19900226)

print('Loading data')
for p, folders, docs in os.walk(data_path):
    for doc in docs:
        with open(os.path.join(p, doc)) as f:
            l = f.readline()
            while l:
                l = l.strip().lower()
                l = l[l.find(' ') + 1:]
                if len(l.split('\t')) == 1:
                    data += [l[:-1].split()]
                l = f.readline()
# Get the original id to word mapping
#    id starts from 1, index 0 is empty
print('Generating idx2word and word2idx')
idx2word = dict(enumerate(set([w for l in data for w in l]), 1))
word2idx = {v: k for k, v in idx2word.items()}

person_name = {'jason', 'jeff', 'julius', 'daniel', 'bernhard', 'emily', 'jessica', 'fred', 'winona','julie','mary','john','greg', 'antoine', 'brian', 'lily', 'sandra', 'yann'}
color_name = {'yellow', 'gray', 'blue', 'red', 'white', 'pink', 'green'}
shape_name = {'rectangle', 'triangle', 'square', 'sphere'}

# the mapping is not perfectly correct
# persons  = [1, 8, 24, 37, 38, 41, 46, 47, 48, 60, 61, 73, 74, 90, 94, 107, 110, 114]
# colors   = [3, 20, 34, 49, 99, 121]
# shapes   = [11, 15, 27, 99]

persons = []
colors = []
shapes = []

for id, word in idx2word.iteritems():
    if word in person_name:
        persons.append(id)
    if word in color_name:
        colors.append(id)
    if word in shape_name:
        shapes.append(id)

print('Repeating person/color/shape')
def repeat_name(l):
    '''
    replace the original person/color/shape to multiple persons/colors/shapes
    :param l:
    :return:
    '''
    ll = []
    for word in l:
        if word2idx[word] in persons:
            k = n_rng.randint(5) + 1
            ll += [idx2word[persons[i]] for i in n_rng.randint(len(persons), size=k).tolist()]
        elif word2idx[word] in colors:
            k = n_rng.randint(5) + 1
            ll += [idx2word[colors[i]] for i in n_rng.randint(len(colors), size=k).tolist()]
        elif word2idx[word] in shapes:
            k = n_rng.randint(5) + 1
            ll += [idx2word[shapes[i]] for i in n_rng.randint(len(shapes), size=k).tolist()]
        else:
            ll += [word]
    return ll
# repeat name of person/color/shape to generate more data? looks not make sense
# data_rep = [repeat_name(l) for l in data]
data_rep = [l for l in data]
# id sequence of data after repeat process
origin   = [[word2idx[w] for w in l] for l in data_rep]

def replace(word):
    if word2idx[word] in persons:
        return '<person>'
    elif word2idx[word] in colors:
        return '<color>'
    elif word2idx[word] in shapes:
        return '<shape>'
    else:
        return word

print('Generating idx2word2 and word2idx2')

# Replace name/color/shape to <tag>
data_clean   = [[replace(w) for w in l] for l in data_rep]
idx2word2    = dict(enumerate(set([w for l in data_clean for w in l]), 1))
idx2word2[0] = '<eol>'
word2idx2    = {v: k for k, v in idx2word2.items()}
Lmax         = len(idx2word2)

for k in xrange(len(idx2word2)):
    print k, '\t', idx2word2[k]
print 'Max: {}'.format(Lmax)
# idx2word(3) and word2idx(4) are from source, size=132
# idx2word2(1) and word2idx2(2) are from target, replacing name/color/shape to <tag>, size=98
print('Exporing dicts to file')
serialize_to_file([idx2word2, word2idx2, idx2word, word2idx], config['voc'])


print('Generating source, target, origin')
# get ready for the dataset.
# source sequence is the sentence with all the real people/colors/shapes converted into tags '<person>', '<color>', '<shape>'
source = [[word2idx2[w] for w in l] for l in data_clean]
# in target, if a word is not person/color/shape, then keep it, else change it to it + Lmax (a word not present in dict)
target = [[word2idx2[w] if w not in ['<person>', '<color>', '<shape>']
           else it + Lmax
           for it, w in enumerate(l)] for l in data_clean]


def print_str(data):
    for d in data:
        print ' '.join(str(w) for w in d)


print_str(data[10000: 10005])
print_str(data_rep[10000: 10005])
print_str(data_clean[10000: 10005])
print_str(source[10000: 10005])
print_str(target[10000: 10005])

serialize_to_file([source, target, origin], config['dataset'])
