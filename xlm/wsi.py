import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.naive_bayes import BernoulliNB
from joblib import Memory
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster.hierarchical
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from pymorphy2 import MorphAnalyzer
import spacy
import string
from collections import Counter
import fire
from xlm.substs_loading import load_substs
from pathlib import Path

################################################################

target_languages = ['english', 'german', 'swedish', 'latin']
corpora_extension = '.txt'
rnc_target_old_positive_words_path = '../data/targets/rumacro_positive.txt'
rnc_target_old_negative_words_path = '../data/targets/rumacro_negative.txt'

def _load_target_words(file):
    with open(str(Path(__file__).parent.absolute()) + '/' + file) as wf:
        target_words = wf.readlines()
    return list(map(lambda x: x.strip(), target_words))

def _get_rumacro_target_words_pair(name):
    name = name.split('_')[0]
    if name == 'rumacroold' or name == 'rumacro':
        return _load_target_words(rnc_target_old_positive_words_path),  _load_target_words(rnc_target_old_negative_words_path)

def load_target_words(name):
    name = name.split('_')[0]
    if name == 'rumacroold' or name == 'rumacro':
        return _load_target_words(rnc_target_old_positive_words_path) + _load_target_words(rnc_target_old_negative_words_path)
    elif name in target_languages:
        return _load_target_words('../data/targets/' + name + '.txt')
    elif name in [i + 'unlem' for i in target_languages]:
        name = name.replace('unlem', '')
        return _load_target_words('data/' + name + '/targets.txt')
    elif 'russe' in name:
        return None
    else:
        assert False, "could not find target words for %s" % name

class Substs_loader:
    def __init__(self, data_name, lemmatizing_method, max_examples=None, delete_word_parts=False,
                 drop_duplicates=True, count_lemmas_weights = False, limit=None):
        self.data_name = data_name
        self.lemmatizing_method = lemmatizing_method
        self.max_examples = max_examples
        self.delete_word_parts = delete_word_parts
        self.drop_duplicates = drop_duplicates
        self.count_lemmas_weights = count_lemmas_weights
        self.translation = str.maketrans('', '', string.punctuation)

        self.dfs = dict()
        self.nf_cnts = dict()
        self.cache = dict()

        if lemmatizing_method is not None and lemmatizing_method!='none':
            if 'ru' in data_name:
                self.analyzer = MorphAnalyzer()
            elif 'german' in data_name:
                self.analyzer = spacy.load("de_core_news_sm", disable=['ner', 'parser'])
            elif 'english' in data_name:
                self.analyzer = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
            else:
                assert "unknown data name %s" % data_name

    def get_nf_cnt(self, substs_probs):
        nf_cnt = Counter(nf for l in substs_probs for p, s in l for nf in self.analyze_russian_word(s))
        return nf_cnt

    def analyze_russian_word(self, word, nf_cnt=None):
        word = word.strip()
        if word not in self.cache:
            self.cache[word] = {i.normal_form for i in self.analyzer.parse(word)}

        if nf_cnt is not None and len(self.cache[word]) > 1:  # select most common normal form
            h_weights = [nf_cnt[h] for h in self.cache[word]]
            max_weight = max(h_weights)
            res = {h for i, h in enumerate(self.cache[word]) if h_weights[i] == max_weight}
        else:
            res = self.cache[word]

        return sorted(list(res))

    def analyze(self, word):
        if not word:
            return ['']

        if not word in self.cache:
            spacyed = self.analyzer(word)
            lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
            self.cache[word] = [lemma]
        return self.cache[word]

    def get_lemmas(self, word, nf_cnt=None):
        if 'ru' in self.data_name:
            return self.analyze_russian_word(word, nf_cnt)
        else:
            return self.analyze(word)

    def get_single_lemma(self, word, nf_cnt):
        return self.get_lemmas(word, nf_cnt)[0]

    def preprocess_substitutes(self, substs_probs, target_word, nf_cnt, topk, exclude_lemmas=set(),
                               delete_word_parts=False):
        """
        1) leaves only topk substitutes without spaces inside
        2) applies lemmatization
        3) excludes unwanted lemmas (if any)
        4) returns string of space separated substitutes
        """
        exclude = exclude_lemmas.union({target_word})

        if delete_word_parts:
            res = [word.strip() for prob, word in substs_probs[:topk] if
                   word.strip() and ' ' not in word.strip() and word[0] == ' ']
        else:
            res = [word.strip() for prob, word in substs_probs[:topk] if
                   word.strip() and ' ' not in word.strip()]

        # TODO: optimise!
        if exclude:
            if self.lemmatizing_method != 'none':
                res = [s for s in res if not set(self.get_lemmas(s)).intersection(exclude)]
            else:
                res = [s for s in res if not s in exclude]

        if self.lemmatizing_method == 'single':
            res = [self.get_single_lemma(word.strip(), nf_cnt) for word in res]
        elif self.lemmatizing_method == 'all':
            res = [' '.join(self.get_lemmas(word.strip(), nf_cnt)) for word in res]
        else:
            assert self.lemmatizing_method == 'none', "unrecognized lemmatization method %s" % self.lemmatizing_method

        return ' '.join(res)

    def get_substitutes(self, path, topk, data_name=None):

        if data_name is None:
            data_name = self.data_name

        if data_name in self.dfs:
            assert data_name in self.nf_cnts
            subst = self.dfs[data_name]
            nf_cnt = self.nf_cnts[data_name]

        else:
            subst = load_substs(path, data_name=data_name, drop_duplicates=self.drop_duplicates,
                                limit=self.max_examples)

            if self.lemmatizing_method != 'none' and self.count_lemmas_weights and 'ru' in self.data_name:
                nf_cnt = self.get_nf_cnt(subst['substs_probs'])
            else:
                nf_cnt = None

            self.dfs[data_name] = subst
            self.nf_cnts[data_name] = nf_cnt

        subst['substs'] = subst.apply(lambda x: self.preprocess_substitutes(x.substs_probs, x.word, nf_cnt,
                                                  topk,delete_word_parts=self.delete_word_parts), axis=1)
        subst['word'] = subst['word'].apply(lambda x: x.replace('ё', 'е'))

        return subst

    def get_substs_pair(self, path1, path2, topk):
        """
        loads subs from path1, path2 and applies preprocessing
        """
        return self.get_substitutes(path1, topk=topk, data_name=self.data_name + '_1'), \
               self.get_substitutes(path2, topk=topk, data_name=self.data_name + '_2' )


def print_mfs_feats(vectorizer, vecs, senses, topf=25):
    mfs_ids = senses.value_counts().index[:2]
    mfs_mask = senses.isin(mfs_ids)

    bnb = BernoulliNB()
    bnb.fit(vecs.astype(np.bool).astype(np.int)[mfs_mask], senses[mfs_mask])
    feature_probs = np.exp(bnb.feature_log_prob_)
    assert feature_probs.shape[
               0] == 2, f'Naive Bayes has weight matrix with {feature_probs.shape[1]} rows! Only 2 rows are supported. Check if it is 2-class data.'
    fn = np.array(vectorizer.get_feature_names())
    result = []
    for cls in range(len(feature_probs)):
        top_feats = feature_probs[cls].argsort()[::-1][:topf]
        result.append(' '.join((f'{feat} {p1:.2f}/{p2:.2f}' for feat, p1, p2 in
                                zip(fn[top_feats], feature_probs[cls, top_feats], feature_probs[1 - cls, top_feats]))))
    return result

def clusterize_search( word, vecs, gold_sense_ids = None ,ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
            affinities=('cosine',), linkages=('average',),
            generate_pictures_df = False,  corpora_ids = None):
    if linkages is None:
        linkages = sklearn.cluster.hierarchical._TREE_BUILDERS.keys()
    if affinities is None:
        affinities = ('cosine', 'euclidean', 'manhattan')
    sdfs = []
    mem = Memory('maxari_cache', verbose=0)
    tmp_dfs = []

    zero_vecs = ((vecs ** 2).sum(axis=-1) == 0)
    if zero_vecs.sum() > 0:
        vecs = np.concatenate((vecs, zero_vecs[:, np.newaxis].astype(vecs.dtype)), axis=-1)

    if generate_pictures_df:
        if gold_sense_ids is not None:
            sense_ids = gold_sense_ids.to_numpy()
            bool_mask = sense_ids[:, None] == sense_ids
        else:
            assert corpora_ids is not None, "gold sense ids and corpora ids are both None"
            w_corpora_ids = corpora_ids
            bool_mask = w_corpora_ids[:, None] == w_corpora_ids

    best_clids = None
    best_silhouette = 0
    distances = []
    for affinity in affinities:
        distance_matrix = cdist(vecs, vecs, metric=affinity)
        distances.append(distance_matrix)

        for nc in ncs:
            for linkage in linkages:
                if linkage == 'ward' and affinity != 'euclidean':
                    continue
                clr = AgglomerativeClustering(affinity='precomputed', linkage=linkage, n_clusters=nc, memory=mem)
                clids = clr.fit_predict(distance_matrix) if nc > 1 else np.zeros(len(vecs))

                ari = ARI(gold_sense_ids, clids) if gold_sense_ids is not None else np.nan
                sil_cosine = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids,metric='cosine')
                sil_euclidean = -1. if len(np.unique(clids)) < 2 else silhouette_score(vecs, clids, metric='euclidean')
                vc = '' if gold_sense_ids is None else '/'.join(
                                        np.sort(pd.value_counts(gold_sense_ids).values)[::-1].astype(str))
                if sil_cosine > best_silhouette:
                    best_silhouette = sil_cosine
                    best_clids = clids

                sdf = pd.DataFrame({'ari': ari,
                                    'word': word, 'nc': nc,
                                    'sil_cosine': sil_cosine,
                                    'sil_euclidean': sil_euclidean,
                                    'vc': vc,
                                    'affinity': affinity, 'linkage': linkage}, index=[0])

                sdfs.append(sdf)

        if generate_pictures_df:
            tmp_df = pd.DataFrame()
            tmp_df['distances'] = distance_matrix.flatten()
            tmp_df['same'] = bool_mask.flatten()
            tmp_df['word'] = word
            w_max_ari = max([i['ari'][0] for i in sdfs if i['word'][0] == word and i['affinity'][0] == affinity])
            tmp_df['ari'] = w_max_ari
            w_max_sil_cosine = max([i['sil_cosine'][0] for i in sdfs if i['word'][0] == word and i['affinity'][0] == affinity])
            tmp_df['sil_cosine'] = w_max_sil_cosine
            tmp_dfs.append(tmp_df)

    picture_df = pd.concat(tmp_dfs) if tmp_dfs else None

    sdf = pd.concat(sdfs, ignore_index=True)
    return best_clids, sdf, picture_df, distances


def test_substs(path1, topk, data_name, lemmatizing_method, max_examples=None, delete_word_parts=True,
                 drop_duplicates=True, count_lemmas_weights = True):
    print(topk, type(topk))
    loader = Substs_loader(data_name=data_name, lemmatizing_method=lemmatizing_method, max_examples=max_examples,
                           delete_word_parts=delete_word_parts,
                 drop_duplicates=drop_duplicates, count_lemmas_weights=count_lemmas_weights)
    s1 = loader.get_substitutes(path1, topk, data_name)

    print(s1.groupby('word').count().to_string())

if __name__=='__main__':
    fire.Fire(test_substs)