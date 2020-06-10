import pandas as pd
import os
import codecs
import re
import pathlib
from pathlib import Path
import csv
from openpyxl import load_workbook
from collections import defaultdict

german_corporas_path = '../data/corporas/'
durel_corpora_names = ['dta18', 'dta19']
surel_corpora_names = ['cook',  'sdewac_1',  'sdewac_2',  'sdewac_3']
durel_target_words_path = 'data/durel_target_words.txt'
surel_target_words_path = 'data/surel_target_words.txt'

durel_dataset_path = '../data/DURel/testset/testset.csv'

target_languages = ['english', 'german', 'swedish', 'latin']

corpora_extension = '.txt'

rnc_dir = '../data/rnc/'
rnc_old_dir = '../data/rnc_old/'
rnc_target_positive_words_path = rnc_dir + 'positive_targets.txt'
rnc_target_negative_words_path = rnc_dir + 'negative_targets.txt'

rnc_target_old_positive_words_path = rnc_old_dir + 'positive_targets.txt'
rnc_target_old_negative_words_path = rnc_old_dir + 'negative_targets.txt'


rnc_parts = {1:'old-1917/', 2:'new1991-/'}
rnc_subdirs = ['positive', 'negative']

################################################################

def _load_target_words(file):
    with open(str(pathlib.Path(__file__).parent.absolute()) + '/' + file) as wf:
        target_words = wf.readlines()
    return list(map(lambda x: x.strip(), target_words))

def _get_rumacro_target_words_pair(name):
    name = name.split('_')[0]
    if name == 'rumacroold':
        return _load_target_words(rnc_target_old_positive_words_path),  _load_target_words(rnc_target_old_negative_words_path)
    if name == 'rumacro':
        return _load_target_words(rnc_target_positive_words_path),  _load_target_words(rnc_target_negative_words_path)

def load_target_words(name):
    name = name.split('_')[0]
    if name == 'rumacroold':
        return _load_target_words(rnc_target_old_positive_words_path) + _load_target_words(rnc_target_old_negative_words_path)
    if name == 'rumacro':
        return _load_target_words(rnc_target_positive_words_path) + _load_target_words(rnc_target_negative_words_path)
    elif name == 'dta':
        return _load_target_words(durel_target_words_path)
    elif name in target_languages:
        return _load_target_words('data/' + name + '/targets.txt')
    elif name in [i + 'unlem' for i in target_languages]:
        name = name.replace('unlem', '')
        return _load_target_words('data/' + name + '/targets.txt')
    elif name in surel_corpora_names:
        return _load_target_words(surel_target_words_path)
    elif 'russe' in name:
        return None
    else:
        assert False, "could not find target words for %s" % name

################################################################
# RNC

def process_rnc_file(lemma, filename):
    wb = load_workbook(filename=filename)
    sheet = wb.active
    tuples = []
    for i, row in enumerate(sheet.rows):
        if i == 0:
            continue
        for i in range(len(row)):
            if row[i].value is None:
                row[i].value = ''
        word = row[3].value[:-1] + row[4].value if row[4].value != '' else row[3].value
        tmp = (row[2].value, word, row[5].value)

        context = r''.join(tmp)
        word = row[3].value.strip()
        word = re.search(r'\b\w+\b', word).group()
        try:
            positions = re.search(r'\b%s\b' % word, r'%s' % context).span()
        except:
            print("Exception on ", context)
            raise()
        #         print(positions)
        new_tuple = {'word': lemma, 'context': context, 'positions': (positions[0], positions[1])}
        #         print(i, new_tuple)
        tuples.append(new_tuple)
    big_df = pd.DataFrame(tuples)
    return big_df

def get_path_word_pairs(dirr, ex, target_words):
    res = []
    for word in target_words:
        path = word + '.xlsx'
        res.append((word, dirr + '/' + ex + '/' + path))
    return res

def load_rnc_corpora_part(dirr, ex, target_words):
    dfs = []
    for word, path in get_path_word_pairs(dirr, ex, target_words):
        dfs.append(process_rnc_file(word, path))
    return pd.concat(dfs, ignore_index = True)

def load_rnc(part, name):
    part_subdir = rnc_parts[int(part)]
    target_words_pair = _get_rumacro_target_words_pair(name)

    dfs = []
    for target_words, subdir in zip(target_words_pair, rnc_subdirs):
        path = str(pathlib.Path(__file__).parent.absolute()) + '/' + rnc_dir + part_subdir
        dfs.append(load_rnc_corpora_part(path, subdir, target_words))

    res = pd.concat(dfs, ignore_index=True)

    # print(res)
    return res, target_words[0] + target_words[1]

################################################################
#Russe

def load_russe_df(part='bts-rnc/train'):
    df = pd.read_csv(Path(__file__).parent/f'russe-wsi-kit/data/main/{part}.csv', sep='\t')

    def parse_positions(s, part):
        a, b = (int(i) for i in s.split('-'))
        return a, b + 1 if 'bts-rnc' in part or 'tax' in part else b

    df['positions'] = df.positions.apply(lambda p: parse_positions(p, part))
    df['word_at'] = df.apply(lambda r: r.context[slice(*r.positions)], axis=1)
    return df

def load_russe_lemm_df(part='bts-rnc/train'):
  df = load_russe_df(part)
  from pymorphy2.tokenizers import simple_word_tokenize
  df['lctx'] = df.apply(lambda r: r.context[:r.positions[0]], axis=1)
  df['rctx'] = df.apply(lambda r: r.context[r.positions[1]:], axis=1)
  from pymorphy2 import MorphAnalyzer
  _ma = MorphAnalyzer()
  _ma_cache = {}
  def ma(s):
    s = s.strip() # get rid of spaces before and after token, pytmorphy2 doesn't work with them correctly
    if s not in _ma_cache:
      _ma_cache[s] = _ma.parse(s)
    return _ma_cache[s]

  def sent_ma(tokens):
    return [ma(t)[0] for t in tokens]

  for col in ('lctx', 'rctx'):
    df[col] = df[col].apply(simple_word_tokenize). \
      apply(sent_ma). \
      apply(lambda l: [s.normal_form for s in l if 'PNCT' not in s.tag]). \
      str.join(' ')
  df.context = df.lctx+' '+df.word+' '+df.rctx
  df.positions = df.apply(lambda r: (len(r.lctx)+1, len(r.lctx)+1+len(r.word)), axis=1)
  df['word_at'] = df.apply(lambda r: r.context[slice(*r.positions)], axis=1)
  return df

################################################################
#From text (DURel, SURel, targets)

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

    # для Procrustus не нужно ничего кроме контекстов.
    if not new_tuples:
        zero_example = dict()
        zero_example['context'] = context
        zero_example['positions'] = None
        zero_example['word'] = None

    return new_tuples

def load_df_from_text(corpora_path, target_words, remove_pos_tags=False):
    with open(corpora_path, 'r') as corpora_file:
        new_tuples = [t for l in corpora_file for t in parse_context(l.strip(), target_words, remove_pos_tags)]

        
    filtered = pd.DataFrame(new_tuples)
    print(filtered['word'].value_counts())

    return filtered

################################################################
#From text (unlemmatized targets)

def get_tokens_matches(seq):
    return [match for match in re.finditer(r'\b[\w]*?\b', seq) if len(match.group().strip())]

def _parse_unlem_context(lemma_context, context, target_words, missed_targets):
    global incorrect_counter
    global incorrect_diff
    lemma_tokens = get_tokens_matches(lemma_context)
    tokens = get_tokens_matches(context)
    new_tuples = []

    correct = True
    if len(lemma_tokens) != len(tokens):
        correct = False

    for lem_tok, tok in zip(lemma_tokens, tokens):
        for target in target_words:
            if target == lem_tok.group():
                example = dict()
                example['context'] = context
                example['positions'] = [tok.start(), tok.end()]
                example['word'] = target
                # example['target'] = target
                if correct:
                    new_tuples.append(example)
                else:
                    missed_targets[target] += 1
    return new_tuples

def parse_unlem_context(lemma_context, context, target_words, missed_targets, remove_pos_tags):
    if remove_pos_tags:
        postfixes = list(set(['_' + i.split('_')[1] for i in target_words]))
        target_words = [i.split('_')[0] for i in target_words]

        for p in postfixes:
            lemma_context.replace(p, '')
    return _parse_unlem_context(lemma_context, context, target_words, missed_targets)

def process_unlem_line(line):
    chars = '.,!?;:'
    for c in chars:
        line = line.replace(" " + c, c)
    line = line.replace("uͤ", "ü")
    line = line.replace("oͤ", "ö")
    line = line.replace("aͤ", "ä")
    line = line.replace("'", "")
    line = line.replace("’", "")
    return line

def load_df_from_unlemmatized_text(lemm_corpora_path, corpora_path, target_words, remove_pos_tags=False):
    missed_targets = defaultdict(int)
    with open(lemm_corpora_path) as fl:
        lines1 = fl.readlines()
    with open(corpora_path) as fl:
        lines2 = fl.readlines()
    tuples = []
    len_tuples = 0
    for line1, line2 in list(zip(lines1, lines2)):
        #     print(line2)
        if 'german' in corpora_path:
            line2 = process_unlem_line(line2)

        res = parse_unlem_context(line1, line2, target_words, missed_targets, remove_pos_tags)
        tuples += res
        if len(tuples) % 300 == 0:
            if len_tuples != len(tuples):
                len_tuples = len(tuples)
                print(len_tuples)

    print("loaded files. Num examples = %d. Missed targets:" % len(tuples))
    print(missed_targets)
    df = pd.DataFrame(tuples)
    return df

################################################################
# General data loading

def load_data(name):
    path_prefix = str(pathlib.Path(__file__).parent.absolute()) + '/'
    target_words = load_target_words(name)

    if 'russe' in name:
        splt = name.split('_')
        ds, part = splt[0], splt[1]
        if ds=='russe' or ds =='russetax':
            df = load_russe_df(part)
        elif ds=='russelemm':
            df = load_russe_lemm_df(part)
        else:
            raise ValueError('Unknown dataset name: ', ds)

    elif 'rumacro' in name:
        part = name.split('_')[1]
        df, target_words = load_rnc(part, name)

    elif name in ['dta_1','dta_2']:
        splt = name.split('_')
        name = durel_corpora_names[int(splt[-1])-1]
        df = load_df_from_text(path_prefix + german_corporas_path + name + corpora_extension, target_words)

    elif name in surel_corpora_names:
        df = load_df_from_text(path_prefix + german_corporas_path + name + corpora_extension, target_words)

    elif name.split('_')[0] in target_languages:
        splt = name.split('_')
        should_clear_pos_tags = splt[0] == 'english'
        path = 'data/' + splt[0] + '/corpus' + splt[1] + '/lemma'
        file = [i for i in os.listdir(path_prefix + path) if '.gz' not in i][0]
        df = load_df_from_text(path_prefix + path + '/' + file, target_words, should_clear_pos_tags)

    elif name.split('_')[0] in [i + 'unlem' for i in target_languages]:
        splt = name.split('_')
        splt[0] = splt[0].replace('unlem', '')
        should_clear_pos_tags = splt[0] == 'english'
        path_lem = 'data/' + splt[0] + '/corpus' + splt[1] + '/lemma'
        file_lem = [i for i in os.listdir(path_lem) if '.gz' not in i][0]
        path = 'data/' + splt[0] + '/corpus' + splt[1] + '/token'
        file = [i for i in os.listdir(path_prefix + path) if '.gz' not in i][0]
        df = load_df_from_unlemmatized_text(path_prefix + path_lem + '/' + file_lem, path_prefix + path + '/' + file,
                                            target_words, should_clear_pos_tags)
    else:
        raise Exception("dataset not recognized - %s\n" % name)

    print('Center words: \n',df.apply(lambda r: r.context[r.positions[0]:r.positions[1]], axis=1).value_counts().to_dict())
    print('Left contexts: \n',df.apply(lambda r: r.context[r.positions[0]-3:r.positions[0]], axis=1).value_counts()[:25].to_dict())
    print('Right contexts: \n',df.apply(lambda r: r.context[r.positions[1]:r.positions[1]+3], axis=1).value_counts()[:25].to_dict())
    df.positions = df.positions.apply(tuple)  # convert list to tuple; to remove duplicates it must be hashable!
    return df, target_words

def test_load_data(name):
    df, _ = load_data(name)
    df1 = df.drop_duplicates(subset=['context','positions'])
    print(f'#ex before/after dropping duplicates: {len(df)}/{len(df1)}')
    df = df1
    print('Column "word":')
    print(df.word.value_counts())
    print('Number of "word" uniq values:', df.word.nunique(), 'number of contexts: ', len(df))
    print(df.context.value_counts())
    sample = df.sample(frac=1).groupby('word').head(500)
    print('sample:', len(sample), len(df))
    print(sample.context.value_counts())

from fire import Fire
if __name__=='__main__':
    Fire(test_load_data)
