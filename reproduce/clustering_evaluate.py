from xlm.data_loading import load_data
#from xlm.sgen_xlm import generate_substitutes
from xlm.substs_loading import load_substs
from collections import defaultdict, Counter
from evaluatable import Evaluatable, GridSearch
from xlm.data_loading import load_data, load_target_words
from xlm.wsi import clusterize_search, get_distances_hist

import sys
import inspect
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import os
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from itertools import product
from joblib import Memory
import fire
import numpy as np
import pandas as pd
import random
from pymorphy2 import MorphAnalyzer
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

def get_word_plot(word, df, output_path, wsi_mode):
    print(word + ' clusters distribution')
    plt.figure(figsize=(10, 8))
    sns.set_context("paper", rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15})

    # width = 0.2
    # ind = np.arange(len(dist1))
    # plt.bar(ind, dist1, width, label='corp_1')
    # plt.bar(ind + width, dist2, width, label='corp_2')
    # plt.xticks(ind + width / 2, ind)

    if wsi_mode:
        df['gold_sense_id'] = df['gold_sense_id'].apply(int)

    hue = 'gold_sense_id' if wsi_mode else 'corpora'
    p = sns.countplot(x="labels", hue=hue, data=df)
    p.tick_params(axis='both', which='major', labelsize=12)

    h, l = p.get_legend_handles_labels()

    plt.legend(h, ['word sense id ' + i for i in l], prop={"size":20})

    img_path = output_path + '/' + word + '-cluster-dist.jpg'
    plt.savefig(img_path)
    return os.path.abspath(img_path).replace("/", "]")
    # img = io.BytesIO()
    # plt.savefig(img)
    # img.seek(0)
    # return img

def get_distances_hist(word, output_path, dist_matrix, mask_scd, bool_mask_wsi, wsi_mode=False):
    plt.figure(figsize=(10, 8))
    sns.set_context("paper", rc={"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 15})

    if wsi_mode:
        ax = sns.distplot(dist_matrix[bool_mask_wsi], label='inside', norm_hist=True)
        ax = sns.distplot(dist_matrix[~bool_mask_wsi], label='between', norm_hist=True)
    else:
        ax = sns.distplot(dist_matrix[mask_scd == 0], label='between', norm_hist=True)
        ax = sns.distplot(dist_matrix[mask_scd == 1], label='old ', norm_hist=True)
        ax = sns.distplot(dist_matrix[mask_scd == 2], label='new', norm_hist=True)

    ax.set_title(word + ' distance histograms')
    # ax.set_xlabel('frequency', size=15)
    # ax.set_ylabel('distance', size=15)

    ax.tick_params(axis='both', which='major', labelsize=12)
    # ax.tick_params(axis='both', which='minor', labelsize=8)

    plt.legend(prop={"size":20})
    # img = io.BytesIO()
    # plt.savefig(img)
    # img.seek(0)
    # return img

    img_path = output_path + '/' + word + '-distance-histogram.jpg'
    plt.savefig(img_path)
    return os.path.abspath(img_path).replace("/", "]")

np.seterr(divide='ignore', invalid='ignore')

class Substs_loader:
    def __init__(self, data_name, lemmatizing_method, topk, max_examples=None, delete_word_parts=False, drop_duplicates=True):
        self.data_name = data_name
        self.lemmatizing_method = lemmatizing_method
        self.topk = topk
        self.max_examples = max_examples
        self.delete_word_parts = delete_word_parts
        self.drop_duplicates = drop_duplicates
        if lemmatizing_method is not None and lemmatizing_method!='none': 
            if 'ru' in data_name:
                self.analyzer = MorphAnalyzer()
            elif 'german' in data_name:
                self.analyzer = spacy.load("de_core_news_sm", disable=['ner', 'parser'])
            elif 'english' in data_name:
                self.analyzer = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
        self.cache = {}

    def analyze(self, word):
        word = word.strip()
        if not word:
            return ['']
        if 'ru' in self.data_name:
            parsed = self.analyzer.parse(word)
            return sorted(list(set([i.normal_form for i in parsed])))
        else:
            spacyed = self.analyzer(word)
            lemma = spacyed[0].lemma_ if spacyed[0].lemma_ != '-PRON-' else spacyed[0].lower_
            return [lemma]

    def get_lemmas(self, word):
        if word not in self.cache:
            self.cache[word] = self.analyze(word)
        return self.cache[word]

    def get_single_lemma(self, word):
        return self.get_lemmas(word)[0]

    def preprocess_substitutes(self, x, exclude_lemmas=[], delete_word_parts=False):
        """
        1) leaves only topk substitutes without spaces inside
        2) applies lemmatization
        3) excludes unwanted lemmas (if any)
        4) returns string of space separated substitutes
        """
        if self.topk is not None:
            x = x[:self.topk]

        if delete_word_parts:
            words = [word.strip() for prob, word in x if word.strip() and ' ' not in word.strip() and word[0] == ' ']
        else:
            words = [word.strip() for prob, word in x if word.strip() and ' ' not in word.strip()]

        if self.lemmatizing_method == 'single':
            words = [self.get_single_lemma(word.strip()) for word in words]
        elif self.lemmatizing_method == 'all':
            words = [' '.join(self.get_lemmas(word.strip())) for word in words]
        else:
            assert self.lemmatizing_method == 'none', "unrecognized lemmatization method %s" % self.lemmatizing_method

        if exclude_lemmas:
            words = [s for s in words if s not in exclude_lemmas]
        return ' '.join(words)

    def get_subs(self, path1, path2):
        """
        loads subs from path1, path2 and applies preprocessing
        """
        subst1 = load_substs(path1, data_name=self.data_name + '_1', drop_duplicates=self.drop_duplicates )
        subst2 = load_substs(path2, data_name=self.data_name + '_2', drop_duplicates=self.drop_duplicates )

        subst1['substs'] = subst1['substs_probs'].apply(lambda x: self.preprocess_substitutes(x, delete_word_parts=self.delete_word_parts))
        subst2['substs'] = subst2['substs_probs'].apply(lambda x: self.preprocess_substitutes(x, delete_word_parts=self.delete_word_parts))

        subst1['word'] = subst1['word'].apply(lambda x: x.replace('ё', 'е'))
        subst2['word'] = subst2['word'].apply(lambda x: x.replace('ё', 'е'))

        if self.max_examples is not None:
            subst1 = subst1.sample(frac=1).groupby('word').head(self.max_examples)
            subst2 = subst2.sample(frac=1).groupby('word').head(self.max_examples)

        return subst1, subst2


class Clustering_Pipeline(Evaluatable):
    def __init__(self, data_name, output_directory, vectorizer_name = 'count_tfidf', min_df = 10, max_df = 0.6,  max_number_clusters = 12,
                 use_silhouette = True, k = 2, n = 5, topk = None, lemmatizing_method = 'none', binary = False,
                 dump_errors = False, max_examples = None, delete_word_parts = False, drop_duplicates=True,
                 path_1 = None, path_2 = None, subst1 = None, subst2 = None, stream=None, should_dump_results=True):
        """
        output_directory -- location where all the results are going to be written

        vectorizer_name - [count, tfidf, count_tfidf]
            count - pure count vectorizer
            tfidf - pure tfidf vectorizer
            count_tfidf - count vectorizer for both subcorpuses, tfidf transformer unique for each subcorpus

        min_df, max_df - vectorizer params

        max_number_clusters - max number of clusters. Ignored when use_silhuette is set,

        use_silhouette - if set, algorithm runs through different number of clusters and pick one
            that gives the highest silhouette score

        k, n - task hyperparameters for binary classification
            word is considered to have gained / lost it's sence if it appears in this sense no more than k times
            in one subcorpus and no less then n times in the other one

        topk - how many (max) substitutes to take for each example

        lemmatizing_method - [none, single, all] - method of substitutes lemmatization
            none - don't lemmatize substitutes
            single - replace each substitute with single lemma (first variant by pymorphy)
            all - replace each substitutete with set of all variants of its lemma (by pymorphy)

        path_1, path_2 - paths to the dumped substitutes files for both corporas. In case you don't want to generate then on the go
        subst1, subst2 - you can also pass the pre-loaded substitutes as dataframes

        """

        print(data_name, vectorizer_name, min_df , max_df,  max_number_clusters ,
                 use_silhouette , k , n , topk , lemmatizing_method, binary,
                 dump_errors , max_examples , delete_word_parts , drop_duplicates,
                 path_1 , path_2 , subst1 , subst2)

        super().__init__(output_directory, should_dump_results)

        self.stream = stream if stream is not None else sys.stdout

        self.data_name = data_name

        self.mem = Memory('clustering_cache', verbose=0)
        self.transformer = None
        self.k = int(k) if int(k) >= 1 else float(k)
        self.n = int(n) if int(n) >= 1 else float(n)
        self.max_number_clusters = max_number_clusters
        self.use_silhouette = use_silhouette
        self.min_df = int(min_df) if int(min_df) >= 1 else float(min_df)
        self.max_df = int(max_df) if int(max_df) >= 1 else float(max_df)

        self.topk = topk

        self.vectorizer_name = vectorizer_name
        self.substitutes_params = dict()

        self.subst1 = subst1
        self.subst2 = subst2

        self.path_1 = path_1
        self.path_2 = path_2
        self.transformer1 = None
        self.transformer2 = None

        self.nonzero_indexes = dict()
        self.distributions = dict()
        self.examples = dict()
        self.contexts = dict()
        self.cluster_most_common = dict()
        self.decision_clusters = dict()

        self.count_vectors = dict()
        self.labels = dict()
        self.sense_ids = dict()
        self.distances = dict()

        self.template = self.path_1.split('/')[-1].split('_')[0]

        self.lemmatizing_method = lemmatizing_method
        self.binary = binary
        self.dump_errors = dump_errors
        self.max_examples = max_examples
        self.delete_word_parts = delete_word_parts
        self.substs_loader = Substs_loader(data_name, lemmatizing_method, topk, max_examples, delete_word_parts, drop_duplicates)

        self.log_df = pd.DataFrame(columns=['word', 'dist1', 'dist2'])

        if vectorizer_name == 'tfidf':
            self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=self.min_df, max_df=max_df,
                                              binary = self.binary)

        elif vectorizer_name in ['count', 'count_tfidf']:
            self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=self.min_df, max_df=max_df,
                                              binary = self.binary)
            if vectorizer_name == 'count_tfidf':
                self.transformer1 = TfidfTransformer()
                self.transformer2 = TfidfTransformer()
        else:
            assert False, "unknown vectorizer name %s" % vectorizer_name

        print(self.get_params())

    def get_params(self):
        """
        return dictionary of hyperparameters to identify the run
        """
        init = getattr(Clustering_Pipeline.__init__, 'deprecated_original', Clustering_Pipeline.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        exclude_params = ['output_directory', 'self','stream', 'subst1', 'subst2', 'needs_preparing']
        parameters = [p.name for p in init_signature.parameters.values()
                      if p.name not in exclude_params and p.kind != p.VAR_KEYWORD]

        values = [getattr(self, key, None) for key in parameters]
        res = dict(zip(parameters, values))
        res['template'] = self.template
        return res

    def get_substs_probs_str(self, substs, limit=50):
        return "; ".join(['"%s" -- %.2f' % (s, p) for p, s in substs[:limit]])

    def get_substs_clean_str(self, substs, limit=50):
        return "; ".join(['"%s"' % s for s in substs[:limit]])

    def show_cluster(self, word, cluster, output, wsi_mode=False):

        clusters_sum1, clusters_sum2, all_sum1, all_sum2 = self.get_sums(*self.labels[word], *self.count_vectors[word])
        dist1, dist2 = self.distributions[word]

        top_words1_pmi = self.get_top_in_clust(all_sum1, clusters_sum1, cluster,
                                               dist1, k=-1) if cluster in clusters_sum1 else []
        top_words2_pmi = self.get_top_in_clust(all_sum2, clusters_sum2, cluster,
                                               dist2, k=-1) if cluster in clusters_sum2 else []
        top_words1_p = self.get_top_p_in_clust(all_sum1, clusters_sum1, cluster,
                                               dist1) if cluster in clusters_sum1 else []
        top_words2_p = self.get_top_p_in_clust(all_sum2, clusters_sum2, cluster,
                                               dist2) if cluster in clusters_sum2 else []

        template = '%.4f %d/%d - <b>"%s"</b>  '

        output.write("<b>Top P subst (sorted by P)</b><br>")
        output.write("<b>Subcorpus 1:</b><br>")
        for p, c1, c2, w in top_words1_p[:15]:
            output.write(template % (p, c1, c2, w))

        if not wsi_mode:
            output.write("<br><b>Subcorpus 2:</b><br>")
            for p, c1, c2, w in top_words2_p[:15]:
                output.write(template % (p, c1, c2, w))

        output.write("<br><br><b>Top-PMI subst first (sorted by PMI)</b><br>")
        output.write("<b>Subcorpus 1:</b><br>")
        for p, c1, c2, w in top_words1_pmi[:15]:
            output.write(template % (p, c1, c2, w))

        if not wsi_mode:
            output.write("<br><b>Subcorpus 2:</b><br>")
            for p, c1, c2, w in top_words2_pmi[:15]:
                output.write(template % (p, c1, c2, w))

        template_both=' P=%.4f, PMI=%.4f %d/%d - <b>"%s"</b>;  '
        output.write("<br><br><b>top-100 substs by P sorted by PMI </b><br>")
        output.write("<b>Subcorpus 1:</b><br>")
        counter = 0

        dct = {w:p for p,_,_,w in top_words1_p}
        print(list(dct.items())[:10])
        print(top_words1_p[:10])
        for pmi, c1, c2, w in top_words1_pmi:
            if w in dct:
                counter += 1
                output.write(template_both % (dct[w], pmi, c1, c2, w))
                if counter >= 15:
                    break

        counter = 0
        if not wsi_mode:
            output.write("<br><b>Subcorpus 2:</b><br>")
            dct = {w: p for p, _, _, w in top_words2_p}
            for pmi, c1, c2, w in top_words2_pmi:
                if w in dct:
                    counter += 1
                    output.write(template_both % (dct[w], pmi, c1, c2, w))
                    if counter >= 15:
                        break

        output.write("<br><br><b>Examples first:</b><br>")
        # print(list(self.examples.keys()))
        # print(list(self.contexts.keys()))
        # print(len(self.examples[word][0][cluster]), self.examples[word][0][cluster])

        n_samples = 10
        length = len(self.contexts[word][0][cluster])
        stride = max(1, length // n_samples + 1 )
        indexes = list(range(0, length, stride))

        # for subs, clean_subs, cont in list(zip(self.examples[word][0][0][cluster], self.examples[word][0][1][cluster], self.contexts[word][0][cluster]))[:10]:
        for index in indexes:
            subs = self.examples[word][0][0][cluster][index]
            clean_subs =self.examples[word][0][1][cluster][index]
            cont = self.contexts[word][0][cluster][index]


            output.write("<b>contexts:</b> " + str(cont) + '<br><b>substs:</b> ' + self.get_substs_probs_str(subs) + '<br>')
            if wsi_mode:
                output.write("<b>gold_sense_id: %d </b><br>" % int(self.sense_ids[word][0][cluster][index]))
            output.write('<br><b>used cleaned substs:</b> ' + self.get_substs_clean_str(clean_subs) + '<br><br>')


        if not wsi_mode:
            output.write("<br><br><b>Examples second:</b><br>")
            n_samples = 10
            length = len(self.contexts[word][1][cluster])
            stride = max(1, length // n_samples)
            indexes = list(range(0, length, stride))
            for index in indexes:
                subs = self.examples[word][1][0][cluster][index]
                clean_subs = self.examples[word][1][1][cluster][index]
                cont = self.contexts[word][1][cluster][index]
                output.write('<b>contexts:</b>' + str(cont) + '<br><b>substs:</b> : ' + self.get_substs_probs_str(subs) + '<br>')
                output.write('<br><b>used cleaned substs:</b> ' + self.get_substs_clean_str(clean_subs) + '<br><br>')

    def analyze_error(self, word, output, label_pairs = None, output_path = '.'):

        # word_plot_dir = output_path + '/word_plot.jpg'
        # dist_hist_path = output_path + '/distance_hist.jpg'
        df = pd.concat([self.subst1[self.subst1['word'] == word], self.subst2[self.subst2['word'] == word]], ignore_index=True)
        wp = get_word_plot(word, df, output_path, label_pairs is None)
        dh = get_distances_hist(word, output_path, *self.distances[word], label_pairs is None)

        # output.write('%s<bs>' % os.path.abspath(word_plot_dir))
        # output.write('%s<bs>' % os.path.abspath(dist_hist_path))
        #
        # output.write('<img src="%s" alt="word plot"><br>' % )
        # output.write('<img src="%s" alt="distance hist plot"><br>' % os.path.abspath(dist_hist_path))
        dist1 = self.distributions[word][0]
        dist2 = self.distributions[word][1]

        if label_pairs is not None:
            predicted, golden = label_pairs[word]
            output.write(word + " : predicted %d, golden %d<br>" % (predicted, golden))
            if predicted == 1:
                cluster = self.decision_clusters[word]
                output.write("decision made on clusted %d<br>" % cluster)
                output.write(
                    "with %d examples in first and %d in second subcorpus<br>" % (dist1[cluster], dist2[cluster]))

        output.write(str(dist1) + '<br>')
        output.write(str(dist2) + '<br>')

        for cluster in range(len(dist1)):
            output.write("<br><b>Cluster %d</b><br><br>" % cluster)
            self.show_cluster(word, cluster, output, label_pairs == None)

        # output.write("<br><br><br><b>Main cluster</b><br>")
        # self.show_cluster(word, main_cluster, output)

        return wp, dh

    def dump_report(self, full_path_dump, label_pairs):
        error_words = []

        if full_path_dump is None:
            output = sys.stdout
        else:
            output = open(full_path_dump, 'w+')
        for w, (i, j) in label_pairs.items():
            if i != j:
                error_words.append(w)
                self.analyze_error(w, output, label_pairs)

        # latex_page = render_to_latex(error_words)
        latex_dump_path = full_path_dump + '.latex'
        with open(latex_dump_path, 'w+') as latex_dump_file:
            latex_dump_file.write(latex_page)

    def _get_score(self, vec1, vec2):
        return cosine(vec1, vec2)

    def _get_vectors(self,word, subs1, subs2):
        #         TRY AND MAKE IT GLOBAL
        #         print((subs1_str.shape, subs2_str.shape))
        self.vectorizer = self.vectorizer.fit(np.concatenate((subs1, subs2)))
        vec1 = self.vectorizer.transform(subs1).todense()
        vec2 = self.vectorizer.transform(subs2).todense()

        vec1_count = vec1
        vec2_count = vec2

        #         TRY AND MAKE IT GLOBAL
        if self.transformer1 is not None and self.transformer2 is not None:
            vec1 = self.transformer1.fit_transform(vec1).todense()
            vec2 = self.transformer2.fit_transform(vec2).todense()

#         vec1_nonzero_mask = ~np.all(np.array(vec1) < 1e-6, axis=1)
#         vec2_nonzero_mask = ~np.all(np.array(vec2) < 1e-6, axis=1)
#         vec1 = vec1[vec1_nonzero_mask]
#         vec2 = vec2[vec2_nonzero_mask]
#         return vec1, vec2, vec1_count, vec2_count, vec1_nonzero_mask, vec2_nonzero_mask

        bool_array_1 = ~np.all(np.array(vec1) < 1e-6, axis=1)
        bool_array_2 = ~np.all(np.array(vec2) < 1e-6, axis=1)
        self.nonzero_indexes[word] = (np.where(bool_array_1)[0], np.where(bool_array_2)[0])

        vec1 = vec1[bool_array_1]
        vec2 = vec2[bool_array_2]

        return vec1, vec2, vec1_count, vec2_count

    def _prepare(self, data_name1, df1, data_name2, df2):
        """
        generate or load substitutes if none provided
        """
        if df1 is not None:
            df1 = df1.dropna(axis=0)
        if df2 is not None:
            df2 = df2.dropna(axis=0)

        if self.path_1 is None and self.subst1 is None:
            self.path_1 = generate_substitutes(data_name=data_name1, dataframe=df1,
                                               **self.substitutes_params)
        if self.path_2 is None and self.subst2 is None:
            self.path_2 = generate_substitutes(data_name=data_name2, dataframe=df2,
                                               **self.substitutes_params)

        if self.subst1 is None or self.subst2 is None:
            self.subst1, self.subst2 = self.substs_loader.get_subs(self.path_1, self.path_2)

        self.subst1["labels"] = np.nan
        self.subst2["labels"] = np.nan

        self.subst1['corpora'] = 'old'
        self.subst2['corpora'] = 'new'

        self.subst1["cluster"] = np.nan
        self.subst2["cluster"] = np.nan

    def save_examples(self, word, subst1, subst2, labels1, labels2, vec1_count, vec2_count):

        self.count_vectors[word] = (vec1_count, vec2_count)
        self.labels[word] = (labels1, labels2)

        subst1 = subst1.iloc[self.nonzero_indexes[word][0]]
        subst2 = subst2.iloc[self.nonzero_indexes[word][1]]

        subst1.reset_index(drop=True, inplace=True)
        subst2.reset_index(drop=True, inplace=True)

        assert len(subst1) == len(labels1) and len(subst2) == len(labels2), "%d %d %d %d" % \
                                                                            (len(subst1), len(labels1), len(subst2),
                                                                             len(labels2))

        cluster_examples1 = defaultdict(list)
        cluster_examples2 = defaultdict(list)

        cluster_contexts1 = defaultdict(list)
        cluster_contexts2 = defaultdict(list)

        cluster_examples_clean1 = defaultdict(list)
        cluster_examples_clean2 = defaultdict(list)

        gold_sence_ids1 =  defaultdict(list)
        gold_sence_ids2 =  defaultdict(list)

        # cluster_most_common1 = defaultdict(str)
        # cluster_most_common2 = defaultdict(str)

        feat_names = self.vectorizer.get_feature_names()

        dump_gold_sence_ids = False
        if 'gold_sense_id' in subst1.keys() and 'gold_sense_id' in subst2.keys():
            dump_gold_sence_ids = True

        for i, l in enumerate(labels1):
            cluster_examples1[l].append(subst1['substs_probs'][i])
            cluster_examples_clean1[l].append([i for i in subst1['substs'][i].split() if i in feat_names])
            cluster_contexts1[l].append(subst1['context'][i])
            if dump_gold_sence_ids:
                gold_sence_ids1[l].append(subst1['gold_sense_id'][i])
            # cluster_most_common1[l] += subst1['substs'][i]

        for i, l in enumerate(labels2):
            cluster_examples2[l].append(subst2['substs_probs'][i])
            cluster_examples_clean2[l].append([i for i in subst2['substs'][i].split() if i in feat_names])
            cluster_contexts2[l].append(subst2['context'][i])
            if dump_gold_sence_ids:
                gold_sence_ids2[l].append(subst1['gold_sense_id'][i])

            # cluster_most_common2[l] += subst2['substs'][i]

        # for key in cluster_most_common1:
        #     cluster_most_common1[key] = Counter(cluster_most_common1[key].split()).most_common()
        #
        # for key in cluster_most_common2:
        #     cluster_most_common2[key] = Counter(cluster_most_common2[key].split()).most_common()

        # self.cluster_most_common[word] = (cluster_most_common1, cluster_most_common2)
        self.examples[word] = ((cluster_examples1, cluster_examples_clean1), (cluster_examples2, cluster_examples_clean2))
        self.contexts[word] = (cluster_contexts1, cluster_contexts2)
        self.sense_ids[word] = (gold_sence_ids1, gold_sence_ids2)

    def get_sums(self, left, right, vec1_count, vec2_count):
        clusters_sum1 = {}
        clusters_sum2 = {}

        for vec, label in zip(vec1_count, left):
            if label not in clusters_sum1:
                clusters_sum1[label] = np.nan_to_num(vec / vec)
            else:
                clusters_sum1[label] += np.nan_to_num(vec / vec)

        for vec, label in zip(vec2_count, right):
            if label not in clusters_sum2:
                clusters_sum2[label] = np.nan_to_num(vec / vec)
            else:
                clusters_sum2[label] += np.nan_to_num(vec / vec)

        all_sum1 = np.zeros(vec1_count[0].shape)
        for vec in vec1_count:
            all_sum1 += np.nan_to_num(vec / vec)

        all_sum2 = np.zeros(vec2_count[0].shape)
        for vec in vec2_count:
            all_sum2 += np.nan_to_num(vec / vec)

        return clusters_sum1, clusters_sum2, all_sum1, all_sum2


    def gen_csv(self, word, left, right, vec1_count, vec2_count, dist1, dist2):

        clusters_sum1, clusters_sum2, all_sum1, all_sum2 = self.get_sums(left, right, vec1_count, vec2_count)
        cols = list()
        vals = list()

        cols.append('word')
        cols.append('dist1')
        cols.append('dist2')

        vals.append(word)
        vals.append(dist1)
        vals.append(dist2)

        labels_unique = list(set(left + right))

        for i in sorted(labels_unique):
            top_words1_pmi = self.get_top_in_clust(all_sum1, clusters_sum1, i,
                                                   dist1) if i in clusters_sum1 else "None"
            top_words2_pmi = self.get_top_in_clust(all_sum2, clusters_sum2, i,
                                                   dist2) if i in clusters_sum2 else "None"
            top_words1_p = self.get_top_p_in_clust(all_sum1, clusters_sum1, i,
                                                   dist1) if i in clusters_sum1 else "None"
            top_words2_p = self.get_top_p_in_clust(all_sum2, clusters_sum2, i,
                                                   dist2) if i in clusters_sum2 else "None"
            contexts1 = self.contexts[word][0][i]
            contexts2 = self.contexts[word][1][i]

            examples1 = self.examples[word][0][i][:10]
            examples2 = self.examples[word][1][i][:10]

            examples1_str = []
            examples2_str = []

            for ex in examples1:
                ex_str = ['%.2f : %s' % i for i in ex[:10]]
                examples1_str.append(ex_str)
            for ex in examples2:
                ex_str = ['%.2f : %s' % i for i in ex[:10]]
                examples2_str.append(ex_str)

            if "{}_dist1_top_words_pmi".format(i) not in self.log_df:
                self.log_df["{}_dist1_top_words_pmi".format(i)] = ""
                self.log_df["{}_dist2_top_words_pmi".format(i)] = ""
                self.log_df["{}_dist1_top_words_p".format(i)] = ""
                self.log_df["{}_dist2_top_words_p".format(i)] = ""
                self.log_df["{}_dist1_contexts".format(i)] = ""
                self.log_df["{}_dist2_contexts".format(i)] = ""
                self.log_df["{}_dist1_substs".format(i)] = ""
                self.log_df["{}_dist2_substs".format(i)] = ""

            cols.append("{}_dist1_top_words_pmi".format(i))
            cols.append("{}_dist2_top_words_pmi".format(i))
            cols.append("{}_dist1_top_words_p".format(i))
            cols.append("{}_dist2_top_words_p".format(i))
            cols.append("{}_dist1_contexts".format(i))
            cols.append("{}_dist2_contexts".format(i))
            cols.append("{}_dist1_substs".format(i))
            cols.append("{}_dist2_substs".format(i))

            vals.append(top_words1_pmi)
            vals.append(top_words2_pmi)
            vals.append(top_words1_p)
            vals.append(top_words2_p)
            vals.append(contexts1)
            vals.append(contexts2)
            vals.append(examples1_str)
            vals.append(examples2_str)

        rows = pd.DataFrame([vals], columns=cols)
        self.log_df = pd.concat([rows, self.log_df])

    def clusterize(self, word, subs1_df, subs2_df):
        """
        clustering.
        subs1 and subs2 dataframes for the specific word on the input
        distributions of clusters for subs1 and subs2 on the output
        """
        subs1 = subs1_df['substs']
        subs2 = subs2_df['substs']

        print("started clustering %s - %d samples" % (word, len(subs1) + len(subs2)), file=self.stream)
        vec1, vec2, vec1_count, vec2_count = self._get_vectors(word, subs1, subs2)
        # vec1_count = vec1
        # vec2_count = vec2
        # vec1_count[vec1_count > 0] = 1
        # vec2_count[vec2_count > 0] = 1

        border = len(vec1)
        transformed = np.asarray(np.concatenate((vec1, vec2), axis=0))

        corpora_ids = np.zeros(len(transformed))
        corpora_ids[border:] = 1

        if self.use_silhouette:
            labels, _, _, distances = clusterize_search(word, transformed, ncs=list(range(2, 5)) + list(range(7, 15, 2)),
            corpora_ids = corpora_ids )
        else:
            labels, _, _, distances = clusterize_search(word, transformed, ncs=(self.max_number_clusters,),
                                             corpora_ids = corpora_ids )

        dist1 = []
        dist2 = []

        left = list(labels[:border])
        right = list(labels[border:])
        for i in sorted(list(set(labels.tolist()))):
            dist1.append(left.count(i))
            dist2.append(right.count(i))

        print(dist1, file=self.stream)
        print(dist2, '\n', file=self.stream)

        distribution_one = np.array(dist1)
        distribution_two = np.array(dist2)

        self.distributions[word] = (distribution_one, distribution_two)

        distance_matrix = distances[0]
        if self.dump_errors:
            labels_mask = np.zeros(distance_matrix.shape)
            labels_mask[:border, :border] = 1
            labels_mask[border:, border:] = 2

            labels_mask[border:, :border] = 3 #to even out counts

            unique, counts = np.unique(labels_mask, return_counts=True)
            # assert max(counts) == counts[0], "%s, shape = %s, border=%d" % (str(list(zip(unique, counts))), distance_matrix.shape, border)

            predicted_labels_mask = labels[:, None] == labels

            self.distances[word] = (distance_matrix, labels_mask, predicted_labels_mask)

        self.distributions[word] = (distribution_one, distribution_two)
        if self.dump_errors:
            self.save_examples(word, subs1_df, subs2_df, left, right, vec1_count, vec2_count)
            # self.gen_csv(word, left, right, vec1_count, vec2_count, dist1, dist2)

        return distribution_one, distribution_two, left, right

    def get_top_in_clust(self, all_sum, clusters_sum, num, clust_size, k=15):
        pmi = list()
        c_clust = list()
        c_all = list()
        for n, score in enumerate(np.asarray(clusters_sum[num])[0]):
            c_word = score
            dataset_size = np.sum(clust_size)
            all_sum = np.asarray(all_sum)
            if clust_size[num] != 0 and all_sum[0][n] != 0:
                pmi.append((c_word / clust_size[num]) / (all_sum[0][n] / dataset_size))
            else:
                pmi.append(0)
            c_clust.append(c_word)
            c_all.append(all_sum[0][n])

        top_ind = np.flip(np.argsort(pmi))
        words = self.vectorizer.get_feature_names()
        top = list()
        for i in top_ind:
            top.append((np.log(pmi[i]), c_clust[i], c_all[i], words[i]))
            k -= 1
            if k == 0:
                break
        return top

    def get_top_p_in_clust(self, all_sum, clusters_sum, num, clust_size, k=15):
        pmi = list()
        c_clust = list()
        c_all = list()
        for n, score in enumerate(np.asarray(clusters_sum[num])[0]):
            c_word = score
            # if score == np.asarray(all_sum)[0][n] and score > clust_size[num]:
            #     print(score, file=self.stream)
            #     print(clusters_sum[num], clust_size[num], file=self.stream)
            dataset_size = np.sum(clust_size)
            all_sum = np.asarray(all_sum)
            if clust_size[num] != 0:
                pmi.append(c_word / clust_size[num])
            else:
                pmi.append(0)
            c_clust.append(c_word)
            c_all.append(all_sum[0][n])

        top_ind = np.flip(np.argsort(pmi))
        words = self.vectorizer.get_feature_names()
        top = list()
        for i in top_ind:
            top.append((pmi[i], c_clust[i], c_all[i], words[i]))
            k -= 1
            if k == 0:
                break
        return top

    def solve(self, target_words, data_name1, df1, data_name2, df2):
        """
        main method
        target words - list of target words
        data_name1, data_name2 - names of the data, such as 'rumacro_1', 'rumacro_2'
        df1, df2 - dataframes with data. Can be None, that data will be loaded using given names
        (or will not be loaded at all if there's no need in generating substitutes)
        """

        self._prepare(data_name1, df1, data_name2, df2)
        distances = []
        binaries = []
        targets = []
        for word in target_words:
            subs1_w = self.subst1[self.subst1['word'] == word]
            subs2_w = self.subst2[self.subst2['word'] == word]
            if len(subs1_w) == 0 or len(subs2_w) == 0:
                print("%s - no samples" % word, file=self.stream)
                continue
            targets.append(word)
            distribution1, distribution2, labels1, labels2 = self.clusterize(word, subs1_w, subs2_w)

            index1 = subs1_w.index[self.nonzero_indexes[word][0]]
            index2 = subs2_w.index[self.nonzero_indexes[word][1]]

            self.subst1.loc[index1, 'predict_sense_id'] = labels1
            self.subst2.loc[index2, 'predict_sense_id'] = labels2

            if distribution1.size == 0 or distribution2.size == 0:
                print("for word %s zero examples in corporas - %d, %d" , file=self.stream%
                      (word, len(distribution1), len(distribution2)))
                distance = sum(distances) / len(distances)
                binary = 1
                distances.append(distance)
                binaries.append(binary)
                continue

            distance = self._get_score(distribution1, distribution2)
            binary = self.solve_binary(word, distribution1, distribution2)
            distances.append(distance)
            binaries.append(binary)

            print(word, ' -- ', distance, ' ', binary, file=self.stream)
        print(len(targets), "words processed", file=self.stream)
        self.log_df.to_csv(r'words_clusters.csv', index=False)

        return list(zip(targets, distances, binaries))

    def solve_binary(self, word, dist1, dist2):
        """
        solving binary classification subtask using the clustering results
        """
        for i, (count1, count2) in enumerate(zip(dist1, dist2)):
            if count1 <= self.k and count2 >= self.n or count2 <= self.k and count1 >= self.n:
                if self.dump_errors:
                    self.decision_clusters[word] = i
                return 1
        return 0

    def solve_for_one_word(self, word, stream = None):
        if stream is not None:
            self.stream = stream
        subs1_w = self.subst1[self.subst1['word'] == word]
        subs2_w = self.subst2[self.subst2['word'] == word]
        if len(subs1_w) == 0 or len(subs2_w) == 0:
            print("%s - no samples<br>" % word, file=self.stream)
            return
        # targets.append(word)
        distribution1, distribution2, labels1, labels2 = self.clusterize(word, subs1_w, subs2_w)

        index1 = subs1_w.index[self.nonzero_indexes[word][0]]
        index2 = subs2_w.index[self.nonzero_indexes[word][1]]

        self.subst1.loc[index1, 'labels'] = labels1
        self.subst2.loc[index2, 'labels'] = labels2

        if distribution1.size == 0 or distribution2.size == 0:
            print("for word %s zero examples in corporas - %d, %d" %
                (word, len(distribution1), len(distribution2)), file=self.stream)
            return None, None

        distance = self._get_score(distribution1, distribution2)
        binary = self.solve_binary(word, distribution1, distribution2)

        print(word, ' -- ', distance, ' ', binary, '<br>',  file=self.stream)
        return binary, distance

###########################################################
"""
hyperparameters that for the search grid
binary parameter 'use_silhuette' is not included as it's processed in a special way
"""
search_ranges = {
    'vectorizer_name' : ['tdidf', 'count', 'count_tfidf'],
    'min_df' : [1, 3, 5, 7, 10, 20, 30],
    'max_df': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    #'k' : [2, 3, 4],
    #'n' : [5, 7, 10],
    'k':list(range(2,6)) + list(range(6,22,3)), #[2,3,4,5,7,10],
    'n':list(range(3,7)) + list(range(7,30,3)),  #[5,7,10,12,15, 20],
    'max_number_clusters': [3, 4, 5, 7, 8, 10, 12, 15],
    'topk' : [15, 30, 50, 100, 150],
    'lemmatizing_method' : ['none', 'single', 'all']
}

class Clustering_Search(GridSearch):
    def __init__(self, output_directory, subst1_path, subst2_path, subdir = None, vectorizer_name = None, min_df = None,
                 max_df = None,  max_number_clusters = None,use_silhouette = None, k = None, n = None,
                 topk = None, lemmatizing_method=None, binary = False, dump_errors = False, max_examples = None,
                 delete_word_parts = False, drop_duplicates=True, should_dump_results=True):
        """
        subst1_path, subst2_path - paths to the substitutes. Two cases:
        1) subst1_path and subst2_path are directories containing substitutes dumps (in that case the search will be
            iterating through all substitutes files found in subst1_path/subdir, considering
            subst2_path/subdir has the same contents)
        2) subst1_path and subst2_path are full paths to the substitutes dump files. 'subdir' param is ignored

        output_directory -- location where all the results are going to be written

        vectorizer_name - [count, tfidf, count_tfidf]
            count - pure count vectorizer
            tfidf - pure tfidf vectorizer
            count_tfidf - count vectorizer for both subcorpuses, tfidf transformer unique for each subcorpus

        min_df, max_df - vectorizer params

        max_number_clusters - max number of clusters. Ignored when use_silhuette is set,

        use_silhouette - if set, algorithm runs through different number of clusters and pick one
            that gives the highest silhouette score

        k, n - task hyperparameters for binary classification
            word is considered to have gained / lost it's sence if it appears in this sense no more than k times
            in one subcorpus and no less then n times in the other one

        topk - how many (max) substitutes to take for each example

        lemmatizing_method - [none, single, all] - method of substitutes lemmatization
            none - don't lemmatize substitutes
            single - replace each substitute with single lemma (first variant by pymorphy)
            all - replace each substitutete with set of all variants of its lemma (by pymorphy)
        """
        super().__init__()
        self.vectorizer_name = vectorizer_name
        self.max_number_clusters = max_number_clusters
        self.use_silhouette = use_silhouette
        self.min_df = min_df
        self.max_df = max_df
        self.k = k
        self.n = n
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
        self.topk = topk
        self.lemmatizing_method = lemmatizing_method
        self.binary = binary
        self.dump_errors = dump_errors
        self.should_dump_results = should_dump_results
        self.template = None
        self.max_examples = max_examples

        self.substitutes_params = dict()
        self.substs = dict()
        self.subst_paths = self.get_subst_paths(subst1_path, subst2_path, subdir)
        self.delete_word_parts = delete_word_parts
        self.drop_duplicates = drop_duplicates

    def get_substs(self, data_name, subst1_path, subst2_path, topk, lemmatizing_method):
        """
        loads substitutes from specific path, unless they already present
        """
        line1 = subst1_path + str(topk) + str(lemmatizing_method)
        line2 = subst2_path + str(topk) + str(lemmatizing_method)

        if line1 not in self.substs or line2 not in self.substs:
            substs_loader = Substs_loader(data_name, lemmatizing_method, topk, self.max_examples,
                                          self.delete_word_parts, drop_duplicates=self.drop_duplicates)
            self.substs[line1], self.substs[line2] = substs_loader.get_subs( subst1_path, subst2_path )

        return self.substs[line1], self.substs[line2]

    def get_subst_paths(self, subst_path1, subst_path2, subdir):
        """
        resolves substitutes path. Returns a list of path pairs to iterate through
        is paths are passed directly to the substitutes dump files, list only contains one element
        """

        if os.path.isfile(subst_path1) or '+' in subst_path1:
            assert os.path.isfile(subst_path2) or '+' in subst_path2, "inconsistent substitutes paths - %s, %s" % (subst_path1, subst_path2)
            self.template = subst_path1.split('/')[-1].split('_')[0]
            return [(subst_path1, subst_path2)]

        tmp_path1 = subst_path1 + '/' + subdir
        tmp_path2 = subst_path2 + '/' + subdir

        files = [i for i in os.listdir(tmp_path1) if i.split('.')[-1] != 'input']
        res = [(tmp_path1 + '/' + file, tmp_path2 + '/' + file) for file in files]
        return res

    def get_output_path(self):
        filename = self.output_directory + '/' + str(self.template) + "_clustering_parameters_search"
        return filename

    def create_evaluatable(self, data_name, params):
        if params['k'] >= params['n']:
            return None
        subst1_path = params['path_1']
        subst2_path = params['path_2']

        substs1, substs2 = self.get_substs(data_name, subst1_path, subst2_path,
                                           params['topk'], params['lemmatizing_method'])

        res = Clustering_Pipeline(data_name, self.output_directory, **params, binary = self.binary,
                                  dump_errors = self.dump_errors, should_dump_results=self.should_dump_results, max_examples = self.max_examples,
                                  delete_word_parts=self.delete_word_parts, drop_duplicates=self.drop_duplicates,
                                  subst1=substs1, subst2=substs2)
        return res

    def _create_params_list(self, lists):
        tmp_res = []
        params_names = list(search_ranges.keys())
        params_names.append('use_silhouette')
        if not self.use_silhouette or self.use_silhouette is None:
            lists['use_silhouette'] = [False]
            tmp_res += [dict(zip(params_names, values)) for values in product(*lists.values())]
        if self.use_silhouette or self.use_silhouette is None:
            lists['use_silhouette'] = [True]
            lists['max_number_clusters'] = [0]
            tmp_res += [dict(zip(params_names, values)) for values in product(*lists.values())]

        res = []
        for subst1_path, subst2_path in self.subst_paths:
            for params in tmp_res:
                params['path_1'] = subst1_path
                params['path_2'] = subst2_path
                res.append(params)
        return res

    def get_params_list(self):
        lists = dict()
        for param, values in search_ranges.items():
            val = getattr(self, param, [])
            assert val != [], "no attribute %s in the class" % param
            if val == None:
                lists[param] = search_ranges[param]
            else:
                lists[param] = [val]
        res = self._create_params_list(lists)
        return res

    def _prepare(self, data_name1, df1, data_name2, df2):
        """
        generates substitutes if necessary to avoid doing it each time in the Evaluatable
        """
        if self.path1 is None and self.subst1 is None:
            self.path1 = generate_substitutes(data_name=data_name1, dataframe=df1.dropna(axis=0),
                                                **self.substitutes_params)
        if self.path2 is None and self.subst2 is None:
            self.path2 = generate_substitutes(data_name=data_name2, dataframe=df2.dropna(axis=0),
                                                **self.substitutes_params)

    def evaluate(self, data_name):
        """
        run the evaluation with given parameters
        checks that all the searchable parameters are set explicitly
        """
        list = self.get_params_list()
        if len(list) > 1:
            nones = []
            for param in search_ranges.keys():
                item = getattr(self, param, None)
                if item is None:
                    nones.append(param)
            print("not all parameters are set: %s" % str(nones))
            return 1
        evaluatable = self.create_evaluatable(data_name, list[0])
        evaluatable.evaluate()

    def solve(self, data_name, output_file_name = None):
        """
        run the evaluation with given parameters
        checks that all the searchable parameters are set explicitly
        """

        list = self.get_params_list()
        if len(list) > 1:
            nones = []
            for param in search_ranges.keys():
                item = getattr(self, param, None)
                if item is None:
                    nones.append(param)
            print("not all parameters are set: %s" % str(nones))
            return 1

        params = list[0]

        list = self.get_params_list()

        if len(list) > 1:
            nones = []
            for param in search_ranges.keys():
                item = getattr(self, param, None)
                if item is None:
                    nones.append(param)
            print("not all parameters are set: %s" % str(nones))
            return 1
        evaluatable = self.create_evaluatable(data_name,params)
        target_words = load_target_words(data_name)
        if target_words is None:
            target_words = evaluatable.subst1['word'].unique()
        else:
            target_words = [i.split('_')[0] for i in target_words]
        evaluatable.solve(target_words,  data_name + '_1', None,  data_name + '_2', None)

        if output_file_name is not None:
            dd = Path(self.output_directory)
            for n,df in enumerate((evaluatable.subst1, evaluatable.subst2)):
                df.loc[df.predict_sense_id.isnull(), 'predict_sense_id'] = -1
                df.to_csv(dd / (output_file_name + f'_{n+1}.csv'), sep='\t', index=False)

if __name__ == '__main__':
    fire.Fire(Clustering_Search)
