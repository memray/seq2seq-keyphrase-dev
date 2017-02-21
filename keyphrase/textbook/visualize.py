import numpy as np
from sklearn.manifold import TSNE

from dataset import keyphrase_test_dataset
from dataset.keyphrase_test_dataset import testing_data_loader
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


config = setup_keyphrase_all()  # load settings.

loader = testing_data_loader('irbooks', kwargs=dict(basedir=config['path']))
docs   = loader.get_docs(return_dict=True)

train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
test_sets = keyphrase_test_dataset.load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config,
                                                                postagging=False)

test_set, test_s_list, test_t_list, test_s_o_list, test_t_o_list, encodings, predictions, scores, idx2word \
    = deserialize_from_file(config['predict_path'] + 'predict.{0}.{1}.pkl'.format(config['predict_type'], 'irbooks'))

book_name_id = {}

for doc, encoding in zip(docs, encodings):
    print('*' * 50)
    # print(doc['name'] + '  -  ' + doc['title'])

    doc['encoding'] = encoding[0][-1][:300]
    doc['book_id']  = doc['name'][:doc['name'].index('_')]

    print(doc['book_id'] + ':' + doc['name'] + '  -  ' + doc['title'])
    # print(doc['encoding'])

    if doc['book_id'] not in book_name_id:
        book_name_id[doc['book_id']] = len(book_name_id)

X = np.asarray([doc['encoding'] for doc in docs])
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
digits_proj = model.fit_transform(X)


def scatter(x, colors):
    colors = np.asarray([book_name_id[doc['book_id']] for doc in docs])
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # # We create a scatter plot.
    # f = plt.figure(figsize=(8, 8))
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

    plt.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    for label, x, y in zip([doc['name']+'-'+doc['title'] for doc in docs], x[:, 0], x[:, 1]):
        print(label)
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


scatter(digits_proj, docs)
plt.savefig(config['predict_path']+'/ir_sections_tsne-generated.png', dpi=1200)