import os

import keyphrase.config as config
from emolga.dataset.build_dataset import deserialize_from_file
from keyphrase.dataset.keyphrase_test_dataset import load_additional_testing_data

if __name__ == '__main__':
    # prepare logging.
    config  = config.setup_keyphrase_all()   # load settings.

    train_set, validation_set, test_sets, idx2word, word2idx = deserialize_from_file(config['dataset'])
    test_sets = load_additional_testing_data(config['testing_datasets'], idx2word, word2idx, config)

    for dataset_name, dataset in test_sets.items():
        print('Exporting %s' % str(dataset_name))

        # keep the first 400 in krapivin
        if dataset_name == 'krapivin':
            dataset['tagged_source'] = dataset['tagged_source'][:400]

        for i, d in enumerate(zip(dataset['tagged_source'], dataset['target_str'])):
            source_postag, target = d
            print('[%d/%d]' % (i, len(dataset['tagged_source'])))

            output_text = ' '.join([sp[0]+'_'+sp[1] for sp in source_postag])

            output_dir = config['baseline_data_path'] + dataset_name + '/text/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_dir+'/'+str(i)+'.txt', 'w') as f:
                f.write(output_text)

            output_text = '\n'.join([' '.join(t) for t in target])
            tag_output_dir = config['baseline_data_path'] + dataset_name + '/keyphrase/'
            if not os.path.exists(tag_output_dir):
                os.makedirs(tag_output_dir)
            with open(tag_output_dir+'/'+str(i)+'.txt', 'w') as f:
                f.write(output_text)
