#!/bin/bash
(export PYTHONPATH=$PYTHONPATH:./; 
	python ./web/app.py \
		--data_name=rushifteval \
		--subst1_path=./combined_old_rse.npz \
		--subst2_path=./combined_new_rse.npz \
		--lemmatizing_method=none \
		--topk=110 \
		--linkage=complete \
		--k=10 \
		--n=15 \
		--mode=scd \
		--vectorizer_name=count \
		--min_df=0.03 \
		--max_df=0.8 \
		--use_silhouette=True \
		--number_of_clusters=0 \
		--drop_duplicates=True \
		--count_lemmas_weights=False \
                --ip="127.0.0.1" \
                --port=5000)
