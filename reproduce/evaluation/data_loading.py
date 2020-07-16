import pandas as pd
import os
import codecs
import re
import pathlib

german_corporas_path = 'data/corporas/'
durel_corpora_names = ['dta18',  'dta19']
surel_corpora_names = ['cook',  'sdewac_1',  'sdewac_2',  'sdewac_3']
durel_target_words_path = 'data/durel_target_words.txt'
surel_target_words_path = 'data/surel_target_words.txt'

durel_dataset_path = 'data/DURel/testset/testset.csv'

target_languages = ['english', 'german', 'swedish', 'latin']

corpora_extension = '.txt'

def load_russe_df(part='bts-rnc/train'):
    df = pd.read_csv(f'russe-wsi-kit/data/main/{part}.csv', sep='\t')

    def parse_positions(s, part):
        a, b = (int(i) for i in s.split('-'))
        return a, b + 1 if 'bts-rnc' in part else b

    df['positions'] = df.positions.apply(lambda p: parse_positions(p, part))
    df['word_at'] = df.apply(lambda r: r.context[slice(*r.positions)], axis=1)
    return df

def parse_context(context, target_words, remove_pos_tags):
    new_tuples = []
    if remove_pos_tags:
        postfixes = list(set(['_' + i.split('_')[1] for i in target_words]))
        target_words = [i.split('_')[0] for i in target_words]

        for p in postfixes:
            context = context.replace(p, '')

    context_lower = context.lower()
    for word in target_words:
        for match in re.finditer(r'\b%s\b' % word.lower(), context_lower):
            assert context_lower[match.start(): match.end()] == word.lower()

            example = dict()
            example['context'] = context
            example['positions'] = (match.start(), match.end())
            example['word'] = word
            new_tuples.append(example)

    return new_tuples


def load_df_from_text(corpora_path, target_words, remove_pos_tags=False):
    with open(corpora_path, 'r') as corpora_file:
        lines = [i.strip() for i in corpora_file.readlines()]

    new_tuples = []
    for line in lines:
        res = parse_context(line, target_words, remove_pos_tags)
        new_tuples += res
        
    filtered = pd.DataFrame(new_tuples)
    print(filtered['word'].value_counts())

    return filtered

def _load_target_words(file):
    with open(file) as wf:
        target_words = wf.readlines()
    return list(map(lambda x: x.strip(), target_words))

def load_data(name):
    path_prefix = str(pathlib.Path(__file__).parent.absolute()) + '/'
    target_words = None
    if 'russe' in name:
        part = name.split('_')[1]
        df = load_russe_df(part)

    elif name in durel_corpora_names or name in ['dta_1','dta_2']:
        splt = name.split('_')
        if len(splt)>1:
            name = durel_corpora_names[int(splt[-1])-1]
        target_words = _load_target_words(path_prefix + durel_target_words_path)
        df = load_df_from_text(path_prefix + german_corporas_path + name + corpora_extension, target_words)

    elif name in surel_corpora_names:
        target_words = _load_target_words(path_prefix + surel_target_words_path)
        df = load_df_from_text(path_prefix + german_corporas_path + name + corpora_extension, target_words)

    elif name.split('_')[0] in target_languages:
        splt = name.split('_')
        path = 'data/' + splt[0] + '/corpus' + splt[1] + '/lemma'
        file = os.listdir(path)[0]

        should_clear_pos_tags = splt[0] == 'english'

        target_words_path = pathlib.Path(__file__).parent.absolute() / 'data' / splt[0] / 'targets.txt'
        target_words = _load_target_words(target_words_path)

        #df = load_df_from_text(path_prefix + path + '/' + file, target_words, should_clear_pos_tags)

    else:
        raise Exception("dataset not recognized - %s\n" % name)

    return target_words


class DURel_loader:
    def __init__(self, filename=str(pathlib.Path(__file__).parent.absolute()) + '/' + durel_dataset_path):
        self.filename = filename
        self.dataset = dict()

    def get_dataset(self):
        if not self.dataset:
            with codecs.open(self.filename, 'r', 'utf-8') as file:
                for line in file:
                    # пропускаем первую строку
                    if u'lemma' in line:
                        continue
                    words = line.split('\t')
                    self.dataset[words[0].strip()] = float(words[7])
        return self.dataset

    def get_binary_dataset(self, threshold):
        dataset = self.get_dataset()
        for key in dataset:
            dataset[key] = 1 if dataset[key] > threshold else 0



