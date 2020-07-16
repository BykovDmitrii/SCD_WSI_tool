import pandas as pd
import os
import codecs
from typing import Dict, List, Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer

from collections import Counter
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import wasserstein_distance
from scipy.stats import spearmanr
import fire
import timeit

from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster.hierarchical
from sklearn.metrics import silhouette_score

from joblib import Memory

from data_loading import load_data, DURel_loader

def get_corpora_id_by_sent_id(sent_id):
    sent_id = sent_id.split('.')
    return int(sent_id[1])


def preprocess_substitutes(x, vectorizer_name, exclude_lemmas=[]):
    words = [word.strip() for prob, word in x]
    if exclude_lemmas:
        words = [s for s in words if not s in exclude_lemmas]

    if vectorizer_name == 'dict':
        return dict(Counter([i.strip() for i in words]))
    else:
        return ' '.join(words)


class SCD_solver:
    def __init__(self, max_number_clusters, min_df, max_df, vectorizer_name='tfidf',
                 disable_tfidf=False, metric='cosine', method='average', use_silhouette=True, language='german'):
        self.max_number_clusters = max_number_clusters
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer_name = vectorizer_name
        self.disable_tfidf = disable_tfidf
        self.metric = metric
        self.method = method
        self.use_silhouette = use_silhouette
        self.language = language

        self.mem = Memory('clustering_cache', verbose=0)

        self.k = 2
        self.n = 5

        if self.language == 'latin':
            self.k = 0
            self.n = 1

    # Just as an example. Could be different metric
    def calculate_distance(self, dist1, dist2):
        return wasserstein_distance(dist1, dist2)

    def clusterize(self, word, substitutes_vector, border):
        global gold_n_senses

        if self.vectorizer_name == 'dict':
            vectorizer = DictVectorizer(sparse=False)
        elif self.vectorizer_name == 'tfidf':
            vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=self.min_df, max_df=self.max_df)
        elif self.vectorizer_name == 'count':
            vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=self.min_df, max_df=self.max_df)
        else:
            assert False, "vectorizer name %s not recognized" % self.vectorizer_name

        #         print(substitutes_vector)

        rep_mat = vectorizer.fit_transform(substitutes_vector).todense()

        if self.vectorizer_name != 'dict_vectorizer' or self.disable_tfidf:
            transformed = rep_mat
        else:
            transformed = TfidfTransformer(norm=None).fit_transform(rep_mat).todense()

        print("started clustering %s - %d samples"%(word, len(substitutes_vector)))

        if self.use_silhouette:
            labels = None
            max_score = 0
            for i, nc in enumerate(list(range(2, 5)) + list(range(7, 15, 2))):
                start = timeit.timeit()

                clr = AgglomerativeClustering(affinity=self.metric, linkage=self.method, n_clusters=nc, memory=self.mem)
                labels_tmp = clr.fit_predict(transformed)
                elapsed = timeit.timeit() - start
                print("clustered. time elapsed - %f" % elapsed)

                score = silhouette_score(transformed, labels_tmp, metric=self.metric)
                if score > max_score:
                    max_score = score
                    labels = labels_tmp
        else:
            clr = AgglomerativeClustering(affinity=self.metric, linkage=self.method,
                                          n_clusters=self.max_number_clusters, memory=self.mem)
            labels = clr.fit_predict(transformed)

        dist1 = []
        dist2 = []
        left = list(labels[:border])
        right = list(labels[border:])
        for i in sorted(list(set(labels.tolist()))):
            dist1.append(left.count(i))
            dist2.append(right.count(i))

        print("labels = %d, border - %d" % (len(labels), border))

        print(labels)
        print(dist1)
        print(dist2, '\n')

        distribution_one = np.array(dist1)
        distribution_two = np.array(dist2)

        return (distribution_one, distribution_two)

    def solve(self, df1, df2, target_words):
        distances = []
        binaries = []
        for word in target_words:
            df1_w = df1[df1['word'] == word]
            df2_w = df2[df2['word'] == word]

            df1_w['substs_probs'] = df1_w['substs_probs'].apply(
                lambda x: preprocess_substitutes(x, self.vectorizer_name, exclude_lemmas=[word]))
            df2_w['substs_probs'] = df2_w['substs_probs'].apply(
                lambda x: preprocess_substitutes(x, self.vectorizer_name, exclude_lemmas=[word]))

            border = len(df1_w)

            substitutes = pd.concat([df1_w['substs_probs'], df2_w['substs_probs']])
            distribution1, distribution2 = self.clusterize(word, substitutes, border)

            if distribution1.size == 0 or distribution2.size == 0:
                print("for word %s zero examples in corporas - %d, %d" %
                      (word, len(distribution1), len(distribution2)))
                distance = sum(distances) / len(distances)
                binary = 1
                distances.append(distance)
                binaries.append(binary)
                continue

            distance = self.calculate_distance(distribution1, distribution2)
            binary = self.solve_binary(distribution1, distribution2)
            distances.append(distance)
            binaries.append(binary)

            print(word, ' -- ', distance, ' ', binary, '\n')
        return distances, binaries

    def run(self, df1, df2, target_words, result_directory='answer/'):
        distances, binaries = self.solve(df1, df2, target_words)
        os.makedirs(result_directory + "task1", exist_ok=True)
        os.makedirs(result_directory + "task2", exist_ok=True)

        with open(result_directory + "task1/" + self.language + '.txt', 'w+') as out:
            for word, score in zip(target_words, binaries):
                out.write("%s\t%s\n" % (word, str(score)))

        with open(result_directory + "task2/" + self.language + '.txt', 'w+') as out:
            for word, score in zip(target_words, distances):
                out.write("%s\t%s\n" % (word, str(score)))

    def solve_binary(self, dist1, dist2):
        for count1, count2 in zip(dist1, dist2):
            if count1 <= self.k and count2 >= self.n:
                return 1
            if count2 <= self.k and count1 >= self.n:
                return 1
        return 0

    def evaluate(self, df1, df2, target_words, golden_scores, output_dir):
        distances = self.run(df1, df2, target_words, output_dir)
        score = spearmanr(distances, golden_scores)
        print("for %d, %f, %f, %s, %d - spearman score = %f\n"% (self.max_number_clusters, self.min_df, self.max_df, self.vectorizer_name, int(self.use_silhouette), score))
        



def get_data(dump1_path, dump2_path):
    corp1 = pd.read_csv(dump1_path+'.input')
    corp2 = pd.read_csv(dump2_path+'.input')
    #corp1, _ = load_data('dta18')
    #corp2, _ = load_data('dta19')

    subs1 = pd.read_csv(dump1_path, index_col=0)['0'].apply(eval)
    subs2 = pd.read_csv(dump2_path, index_col=0)['0'].apply(eval)

    subs1.reset_index(drop=True, inplace=True)
    corp1['substs_probs'] = subs1
    subs2.reset_index(drop=True, inplace=True)
    corp2['substs_probs'] = subs2

    return corp1, corp2


def get_dump_files(directory, language):
    masks = {'english' : '-or-T', 'german' : '-oder-T', 'swedish' : '-eller-T', 'latin' : '-aut-T' }

    tmp1 = directory + '/' + language + '_1-limitNone'
    tmp2 = directory + '/' + language + '_2-limitNone'

    tmp1 += '/' + os.listdir(tmp1)[0]
    tmp2 += '/' + os.listdir(tmp2)[0]

    file1 = [i for i in os.listdir(tmp1) if masks[language] in i and 'input' not in i][0]
    file2 = [i for i in os.listdir(tmp2) if masks[language] in i and 'input' not in i][0]

    res1 = tmp1 + '/' + file1
    res2 = tmp2 + '/' + file2

    print (res1)
    print (res2)

    return res1, res2

def run_new(directory, language):

    dump1_path, dump2_path = get_dump_files(directory, language)

    metric = 'cosine'
    method = 'average'
    vec = 'tfidf'

    corp1, corp2 = get_data(dump1_path, dump2_path)

    target_words=list(corp1['word'].unique())

    for min_df in [0.01, 0.02, 0.03]:
	    max_df = 0.98
	    use_silhouette = True

	    line = '_'.join([vec, str(min_df), str(max_df), 'silh_subst12']) + '/'
	    solver = SCD_solver(4, min_df, max_df, vectorizer_name=vec,
				metric=metric, method=method, use_silhouette=use_silhouette, language=language)
	    solver.run(corp1, corp2, target_words, line)

if __name__ == '__main__':
	fire.Fire(run_new)
