from pathlib import Path
import pandas as pd
import numpy as np 
from time import time
from collections import Counter
import os
import regex as re
from sklearn.feature_extraction import DictVectorizer

def intersect_sparse(substs_probs, substs_probs_y, nmasks=1, s=0):
    print('intersect_sparse, arg1:', substs_probs.iloc[0], 'intersect_sparse, arg2:', substs_probs_y.iloc[0], sep='\n')
    vec = DictVectorizer(sparse=True)
    f1=substs_probs.apply(lambda l: {s:p for p,s in l})
    f2=substs_probs_y.apply(lambda l: {s:p for p,s in l})
    vec.fit(list(f1)+list(f2))
    f1,f2 = (vec.transform(list(f)) for f in (f1,f2))
    alpha1, alpha2 = ( (1. - f.sum(axis=-1).reshape(-1,1)) / 250000**nmasks for f in (f1, f2) )
    prod = f1.multiply(f2) + f1.multiply(alpha2) + f2.multiply(alpha1)

    fn = np.array(vec.feature_names_)
    m = prod
    idx = ( [ m.indices[m.indptr[i]:m.indptr[i+1]][ np.argsort(m.data[m.indptr[i]:m.indptr[i+1]])[::-1] ] for i in range(m.shape[0]) ] )
    l=[[(p,s) for p,s in zip(prod[i].toarray()[0,jj],fn[jj]) if s.startswith(' ') and ' ' not in s.strip()]  for i,jj in enumerate(idx)]    
    print('Combination: ', l[:3])
    return l

def bcomb3(df, nmasks=1, s=0):
    vec = DictVectorizer(sparse=True)
    f1=df.substs_probs.apply(lambda l: {s:p for p,s in l})
    f2=df.substs_probs_y.apply(lambda l: {s:p for p,s in l})
    vec.fit(list(f1)+list(f2))
    f1,f2 = (vec.transform(list(f)) for f in (f1,f2))
    for f in (f1,f2):
        f += (1. - f.sum(axis=-1).reshape(-1,1)) / 250000**nmasks
    log_prod = np.log(f1)+np.log(f2)
    log_prior=np.log( (f1.mean(axis=0)+f2.mean(axis=0))/2 )
    fpmi = log_prod - s*log_prior

    fn = np.array(vec.feature_names_)
    maxlen=(df.substs_probs_y.apply(len)+ df.substs_probs.apply(len)).max()
    idx = fpmi.argsort(axis=-1)[:,:-maxlen-1:-1]
    l=[[(p,s) for p,s in zip(fpmi[i,jj],fn[jj]) if s.startswith(' ') and ' ' not in s.strip()]  for i,jj in enumerate(idx)]    
    df['substs_probs'] = l
    return df

def load_substs(substs_fname, limit=None, drop_duplicates=True, data_name = None, drop_word_groups=True):
    if substs_fname.endswith('&'):
        split = substs_fname.strip('&').split('&')
        print(f'Combining:', split)
        dfinps = [load_substs_(p, limit, drop_duplicates, data_name) for p in split]
        res = dfinps[0]
        nm = len(split[0].split('<mask>'))-1
        for dfinp in dfinps[1:]:
            res = res.merge(dfinp, on=['context','positions'], how='inner', suffixes=('','_y'))
            res.substs_probs = intersect_sparse(res.substs_probs, res.substs_probs_y, nmasks=nm, s=0.0)        
            res.drop(columns=[c for c in res.columns if c.endswith('_y')], inplace=True)
        return res
    elif substs_fname.endswith('+'):
        split = substs_fname.strip('+').split('+')
        p1 = '+'.join(split[:-1])
        s = float(split[-1]) 
        p2 = re.sub(r'((<mask>)+)(.*?)T',r'T\3\1',p1)
        if p2==p1:
            p2 =  re.sub(r'T(.*?)((<mask>)+)',r'\2\1T',p1)
        print(f'Combining {p1} and {p2}')
        if p1==p2:
            raise Exception('Cannot conver fname to symmetric one:', p1)
        dfinp1, dfinp2 = (load_substs_(p, limit, drop_duplicates, data_name) for p in (p1,p2))
        dfinp = dfinp1.merge(dfinp2, on=['context','positions'], how='inner', suffixes=('','_y'))
        dfinp.substs_probs = intersect_sparse(dfinp.substs_probs, dfinp.substs_probs_y, nmasks=len(substs_fname.split('<mask>'))-1, s=s)
        dfinp.drop(columns=[c for c in dfinp.columns if c.endswith('_y')], inplace=True)
        return dfinp
    else:
        return load_substs_(substs_fname, limit, drop_duplicates, data_name, drop_word_groups)

def load_substs_(substs_fname, limit=None, drop_duplicates=True, data_name = None, drop_word_groups=True):
    st = time()
    p = Path(substs_fname)
    npz_filename_to_save = None
    print(time()-st, 'Loading substs from ', p)
    if substs_fname.endswith('.npz'):
        arr_dict = np.load(substs_fname, allow_pickle=True)
        ss,pp = arr_dict['substs'], arr_dict['probs']
        print(ss.shape, ss.dtype, pp.shape, pp.dtype)
        ss,pp = [list(s) for s in ss], [list(p) for p in pp]
        substs_probs = pd.DataFrame({'substs':ss, 'probs':pp})
        substs_probs = substs_probs.apply(lambda r: [(p,s) for s,p in zip(r.substs, r.probs)], axis=1)
    else:
        substs_probs = pd.read_csv(p, index_col=0, nrows=limit)['0']

        print(time()-st, 'Eval... ', p)
        substs_probs = substs_probs.apply(pd.eval)
        print(time()-st, 'Reindexing... ', p)
        substs_probs.reset_index(inplace = True, drop = True)

        szip = substs_probs.apply(lambda l: zip(*l)).apply(list)
        res_probs, res_substs = szip.str[0].apply(list), szip.str[1].apply(list)

        npz_filename_to_save = p.parent/(p.name.replace('.bz2', '.npz'))
        if not os.path.isfile(npz_filename_to_save):
            print('saving npz to %s' % npz_filename_to_save)
            np.savez_compressed(p.parent/(p.name.replace('.bz2', '.npz')), probs=res_probs, substs=res_substs)

    p_ex = p.parent / (p.name+'.input')
    assert(os.path.isfile(p_ex))
    print(time()-st,'Loading examples from ', p_ex)
    dfinp = pd.read_csv(p_ex, nrows=limit)
    dfinp['positions'] = dfinp['positions'].apply(pd.eval)
    dfinp['word_at'] = dfinp.apply(lambda r: r.context[slice(*r.positions)], axis=1)

    dfinp.positions = dfinp.positions.apply(tuple)
    dfinp['substs_probs'] = substs_probs
    if drop_duplicates:
        dfinp = dfinp.drop_duplicates('context')
    if drop_word_groups:
        dfinp['substs_probs'] = dfinp['substs_probs'].apply(lambda x: [(p, s) for p, s in x if s.startswith(' ') and ' ' not in s.strip()])

    dfinp.reset_index(inplace = True)
    print(dfinp.head())
    dfinp['positions'] = dfinp.positions.apply(tuple)
    return dfinp

def sstat(substs_fname, limit=None):
    dfinp = load_substs(substs_fname, limit)
    print('Counting stat')
    print(dfinp.head(3))
    dfinp['word_at'] = dfinp.apply(lambda r: r.context[r.positions[0]: r.positions[1]], axis=1)
    rdf = dfinp.groupby('word').agg({
                'word_at':lambda x:Counter(x).most_common(), 
                'substs_probs':lambda x:Counter(s for l in x for p,s in l).most_common()}).reset_index()

    print(dfinp.groupby('word').count().to_string())
    stats_fname = substs_fname + '.sstat.tsv'
    with open(stats_fname, 'w') as outp:
        for i,r in rdf.iterrows():
            print(r.word, ','.join('%s %d' % p for p in r.word_at), file=outp)
            print('->', ','.join('%s %d' % p for p in r.substs_probs), file=outp)
    print('Stats saved to ', stats_fname)

from fire import Fire
if __name__=='__main__':
    Fire(sstat)
