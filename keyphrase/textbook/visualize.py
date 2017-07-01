import numpy as np
from sklearn.manifold import TSNE

from keyphrase.dataset import keyphrase_test_dataset
from keyphrase.dataset.keyphrase_test_dataset import testing_data_loader
from emolga.dataset.build_dataset import deserialize_from_file, serialize_to_file
from keyphrase.config import *

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def chapter_scatter(x):
    colors = np.asarray([book_name_id[doc['book_id']] for doc in docs])
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # # We create a scatter plot.
    f = plt.figure(figsize=(20, 20))
    # ax = plt.subplot(aspect='equal')
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
    #                 c=palette[colors.astype(np.int)])
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    # ax.axis('tight')
    #
    # # We add the labels for each digit.
    # txts = []
    # for i in range(len(colors)):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    for label, x, y in zip([doc['name'] + '-' + doc['title'] for doc in docs], x[:, 0], x[:, 1]):
        label = str(label)
        if not label.startswith('mir') and not label.startswith('iir'):
            continue

        print(label)
        book_name = label[:label.index('_')]
        begin_index = label.find('_') + 1
        end_index = label.find('_', label.index('_') + 1)
        if end_index == -1:
            end_index = label.find('-')
        chapter_number = int(label[begin_index: end_index])
        print(book_name + '-' + str(chapter_number))

        if book_name=='mir':
            if chapter_number < 2 or chapter_number > 8:
                continue
            color = 'r'

        if book_name=='iir':
            if chapter_number < 1 or chapter_number > 12:
                continue
            color = 'g'

        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.scatter(x, y, lw=0, s=40, c=color)

    plt.savefig(config['predict_path']+'/ir_sections_tsne-generated.png', dpi=120)
    plt.show()


def plot_chapter_vectors():
    for doc, encoding in zip(docs, input_encodings):
        print('*' * 50)
        print(doc['name'] + '  -  ' + doc['title'])

        doc['forward_encoding'] = encoding[0][-1][:300]
        doc['backward_encoding'] = encoding[0][0][300:]
        doc['book_id']  = doc['name'][:doc['name'].index('_')]

        # print(doc['book_id'] + ':' + doc['name'] + '  -  ' + doc['title'])
        # print(doc['encoding'])

        if doc['book_id'] not in book_name_id:
            book_name_id[doc['book_id']] = len(book_name_id)

    # serialize_to_file(docs, config['path'] + '/dataset/textbook_linking/docs.pkl')

    X = np.asarray([doc['forward_encoding'] for doc in docs])
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    digits_proj = model.fit_transform(X)

    chapter_scatter(digits_proj)


def cut_zero(sample_index, idx2word):
    sample_index = list(sample_index)
    # if 0 not in sample:
    #     return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample]
    # # return the string before 0 (<eol>)
    # return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample[:sample.index(0)]]

    if 0 in sample_index:
        sample_index = sample_index[:sample_index.index(0)]

    wordlist = []
    find_copy = False
    for w_index in sample_index:
        if w_index >= config['voc_size']:
            find_copy = True
        else:
            wordlist.append(idx2word[w_index].encode('utf-8'))
    if find_copy:
        # print('Find copy! - %s - %s' % (' '.join(wordlist), str(sample_index)))
        wordlist = None
    return wordlist

def phrase_scatter(x, labels):
    # # We create a scatter plot.
    f = plt.figure(figsize=(20, 20))

    for label, x, y in zip(labels, x[:, 0], x[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.scatter(x, y, lw=0, s=40, c='black')

    plt.savefig(config['predict_path']+'/ir_phrase_tsne-generated.png', dpi=120)
    plt.show()


def plot_phrase_vectors():

    phrase_dict = {}
    for doc, prediction_list, score_list, encoding_list in zip(docs, predictions, scores, output_encodings):
        # print('*' * 50)
        # print(doc['name'] + '  -  ' + doc['title'])

        doc['book_id']  = doc['name'][:doc['name'].index('_')]

        number_to_keep = 10
        for prediction, score, encoding in zip(prediction_list, score_list, encoding_list):

            predicted_word = cut_zero(prediction, idx2word)
            if predicted_word == None:
                continue
            # if len(predicted_word)==1:
            #     continue

            if ' '.join(predicted_word) not in phrase_dict:
                phrase_dict[' '.join(predicted_word)] = {'score':score, 'encoding':encoding, 'word':' '.join(predicted_word), 'times':1}
            else:
                if score < phrase_dict[' '.join(predicted_word)]['score']:
                    phrase_dict[' '.join(predicted_word)]['score'] = score
                    phrase_dict[' '.join(predicted_word)]['encoding'] = encoding
                phrase_dict[' '.join(predicted_word)]['times']+=1

            number_to_keep -= 1
            if number_to_keep == 0:
                break

    for p in phrase_dict.values():
        p['score']/=p['times']

    K = 200
    p_list = sorted(phrase_dict.values(), key=lambda x:x['score']) #[:K]
    print('#(phrase)=%d' % len(phrase_dict))
    X = np.asarray([phrase['encoding'] for phrase in p_list])
    label = np.asarray([phrase['word'] for phrase in p_list])
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    digits_proj = model.fit_transform(X)

    phrase_scatter(digits_proj, label)

if __name__=='__main__':
    config = setup_keyphrase_all()  # load settings.

    loader = testing_data_loader('irbooks', kwargs=dict(basedir=config['path']))
    docs   = loader.get_docs(return_dict=True)

    train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    test_sets = keyphrase_test_dataset.load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config,
                                                                    postagging=False)

    test_set, test_s_list, test_t_list, test_s_o_list, test_t_o_list, input_encodings, predictions, scores, output_encodings, idx2word \
        = deserialize_from_file(config['predict_path'] + 'predict.{0}.{1}.pkl'.format(config['predict_type'], 'irbooks'))

    book_name_id = {}

    # plot_phrase_vectors()
    plot_chapter_vectors()