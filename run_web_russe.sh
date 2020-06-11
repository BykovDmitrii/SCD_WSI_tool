#!/bin/bash
(export PYTHONPATH=$PYTHONPATH:./; 
	python ./web/app.py \
		--data_name=russe_bts-rnc/train \
		--output_directory=bts_rnc_res \
		--subst1_path=$1 \
		--lemmatizing_method=all \
		--topk=150 \
		--k=2 \
		--n=7 \
		--vectorizer_name=count \
		--min_df=0.03 \
		--max_df=0.8 \
		--use_silhouette=False \
		--max_number_clusters=3 \
		--drop_duplicates=True \
		--count_lemmas_weights=True)
