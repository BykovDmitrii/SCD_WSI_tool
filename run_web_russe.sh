#!/bin/bash
(export PYTHONPATH=$PYTHONPATH:./; 
	python ./web/app.py \
		--data_name=russe_bts-rnc/train \
		--output_directory=bts_rnc_res \
		--subst1_path=$1/\<mask\>\<mask\>-\(а-также-T\)-2ltr2f_topk150_fixspacesTrue.npz+0.0+ \
		--lemmatizing_method=all \
		--topk=150 \
		--k=2 \
		--n=7 \
		--vectorizer_name=count \
		--min_df=0.03 \
		--max_df=0.8 \
		--use_silhouette=False \
		--number_of_clusters=3 \
		--drop_duplicates=True \
		--count_lemmas_weights=True \
                --ip="$2" \
                --port=$3)
