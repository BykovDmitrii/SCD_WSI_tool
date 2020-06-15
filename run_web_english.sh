#!/bin/bash
(export PYTHONPATH=$PYTHONPATH:./; 
	python ./web/app.py \
		--data_name=english \
		--subst1_path=$1/\<mask\>\<mask\>-or-T-2ltr2f_topk200_fixspacesTrue.npz+0.0+ \
		--subst2_path=$2/\<mask\>\<mask\>-or-T-2ltr2f_topk200_fixspacesTrue.npz+0.0+ \
		--lemmatizing_method=none \
		--topk=150 \
		--k=10 \
		--n=15 \
		--vectorizer_name=count \
		--min_df=0.03 \
		--max_df=0.8 \
		--use_silhouette=True \
		--number_of_clusters=0 \
		--drop_duplicates=True \
		--count_lemmas_weights=False \
                --ip="$3"
                --port=$4)
