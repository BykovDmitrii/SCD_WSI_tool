from xlm.substs_loading import load_substs
from collections import Counter
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import fire
from xlm.wsi import clusterize_search, get_distances_hist, Substs_loader
import numpy as np
from pymorphy2 import MorphAnalyzer

_ma = MorphAnalyzer()
_ma_cache = {}

def ma(s):
    # return [ww(s.strip())]
    s = s.strip()  # get rid of spaces before and after token, pytmorphy2 doesn't work with them correctly
    if s not in _ma_cache:
        _ma_cache[s] = _ma.parse(s)
    return _ma_cache[s]


def get_normal_forms(s, nf_cnt=None):
    hh = ma(s)
    if nf_cnt is not None and len(hh) > 1:  # select most common normal form
        h_weights = [nf_cnt[h.normal_form] for h in hh]
        max_weight = max(h_weights)
        return {h.normal_form for i, h in enumerate(hh) if h_weights[i] == max_weight}
    else:
        return {h.normal_form for h in hh}

def max_ari(df, X, ncs=list(range(1, 5, 1)) + list(range(5, 12, 2)),
            affinities=('cosine',), linkages=('average',), vectorizer=None, print_topf=None,
            pictures_dir = None, pictures_prefix = None):

    if pictures_dir is not None:
        os.makedirs(pictures_dir, exist_ok=True)
    sdfs = []
    picture_dfs = []
    for word in df.word.unique():
        mask = (df.word == word)
        vectors = X[mask] if vectorizer is None else vectorizer.fit_transform(X[mask]).toarray()
        gold_sense_ids = df.gold_sense_id[mask]

        _, sdf, picture_df, _ = clusterize_search(word, vectors, gold_sense_ids,ncs=ncs,
                affinities=affinities, linkages=linkages, print_topf=print_topf,
                generate_pictures_df = pictures_dir is not None)
        sdfs.append(sdf)
        if picture_df is not None:
            picture_dfs.append(picture_df)

    picture_df = pd.concat(picture_dfs, ignore_index=True) if picture_dfs else None
    if pictures_dir is not None:
        assert pictures_prefix is not None
        assert not picture_df is not None

        path1 = pictures_dir + '/' + pictures_prefix + 'separate.svg'
        path2 = pictures_dir + '/' + pictures_prefix + 'all.svg'

        get_distances_hist(picture_df, path1, path2)

    sdf = pd.concat(sdfs, ignore_index=True)
    # groupby is docuented to preserve inside group order
    res = sdf.sort_values(by='ari').groupby(by='word').last()
    # maxari for fixed hypers
    fixed_hypers = sdf.groupby(['affinity', 'linkage', 'nc']).agg({'ari': np.mean}).reset_index()
    idxmax = fixed_hypers.ari.idxmax()
    res_df = fixed_hypers.loc[idxmax:idxmax].copy()
    res_df = res_df.rename(columns=lambda c: 'fh_maxari' if c == 'ari' else 'fh_' + c)
    res_df['maxari'] = res.ari.mean()

    for metric in [c for c in sdf.columns if c.startswith('sil')]:
        res_df[metric+'_ari'] = sdf.sort_values(by=metric).groupby(by='word').last().ari.mean()

    return res_df, res, sdf


def combine(substs_probs1, substs_probs2):
    spdf = substs_probs1.to_frame(name='sp1')
    spdf['sp2'] = substs_probs2
    spdf['s2-dict'] = spdf.sp2.apply(lambda l: {s: p for p, s in l})
    res = spdf.apply(lambda r: sorted([(p * r['s2-dict'][s], s) for p, s in r.sp1 if s in r['s2-dict']], reverse=True,
                                      key=lambda x: x[0]), axis=1)
    return res

def do_run_max_ari(substitutes_dump, data_name, topk=None, vectorizer=None, min_df=None, max_df=None,
                   dump_images=False):

    # df = load_substs(substitutes_dump, data_name=data_name)
    # nf_cnt = get_nf_cnt(df['substs_probs'])

    loader = Substs_loader(data_name, lemmatizing_method='all', drop_duplicates=False, count_lemmas_weights=True)

    dump_directory = substitutes_dump + '_dump'
    sdfs = []
    exclude = []
    vec_names = ['TFIDF_Vectorizer', 'CountVectorizer']
    vectorizers = [TfidfVectorizer, CountVectorizer] if vectorizer is None else [eval(vectorizer)]
    mindfs = [0.05, 0.03, 0.02, 0.01, 0.0] if min_df is None else [min_df]
    maxdfs = [0.98, 0.95, 0.9, 0.8] if max_df is None else [max_df]
    topks = 2**np.arange(8, 3, -1) if topk is None else [topk]

    for topk in topks:
        df = loader.get_substitutes(substitutes_dump, topk)
        substs_texts = df['substs']
        # substs_texts = df.apply(lambda r: preprocess_substs(r.substs_probs[:topk], nf_cnt=nf_cnt, lemmatize=True,
        #                                                     exclude_lemmas=exclude + [r.word]), axis=1).str.join(' ')
        for vec_id, vec_cls in enumerate(vectorizers):
            local_dump_dir = '/'.join([dump_directory, str(topk), vec_names[vec_id]])
            os.makedirs(local_dump_dir, exist_ok=True)

            for min_df in mindfs:
                for max_df in maxdfs:
                    dump_filename_prefix = '/%d_%s_%f_%f' % (topk, vec_names[vec_id], min_df, max_df)
                    dump_path_prefix =  local_dump_dir + dump_filename_prefix

                    pictures_dir = local_dump_dir if dump_images else None
                    pictures_prefix = dump_filename_prefix if dump_images else None

                    vec = vec_cls(token_pattern=r"(?u)\b\w+\b", min_df=min_df, max_df=max_df)
                    res_df, res, sdf = max_ari(df, substs_texts, affinities=('cosine',), linkages=('average',),
                                            vectorizer=vec, pictures_dir = pictures_dir, pictures_prefix = pictures_prefix)
                    res_df = res_df.assign(topk=topk, vec_cls=vec_cls.__name__, min_df=min_df, max_df=max_df)
                    sdfs.append(res_df)

                    print(dump_path_prefix, max(res_df['maxari']))

                    dump_filename = dump_path_prefix + '.res'
                    res.to_csv(dump_filename)

                    dump_filename = dump_path_prefix + '.res_df'
                    res_df.to_csv(dump_filename)

                    dump_filename = dump_path_prefix + '.sdf'
                    sdf.to_csv(dump_filename)

    sdfs_df = pd.concat(sdfs, ignore_index=True)
    dump_filename = dump_directory + '/dump_general.sdfs'
    for metric in ['maxari','fh_maxari', 'sil_cosine_ari', 'sil_euclidean_ari']:
        if metric not in sdfs_df.columns:
            continue
        res_fname = dump_filename+'.'+metric
        sdfs_df.sort_values(by=metric, ascending=False).to_csv(res_fname, sep='\t')
        # print('Saved results to:\n', res_fname)
        # print(metric, sdfs_df[metric].describe())
    # pd.concat(sdfs, ignore_index=True).sort_values(by='fh_maxari')
    print(pd.concat(sdfs, ignore_index=True).sort_values(by='maxari', ascending=False).head(5).to_string())
    # print(pd.concat(sdfs, ignore_index=True).sort_values(by='maxari', ascending=False)['maxari'])
    # pd.concat(sdfs, ignore_index=True).sort_values(by='fh_maxari').tail(25)

if __name__ == '__main__':
    fire.Fire(do_run_max_ari)
