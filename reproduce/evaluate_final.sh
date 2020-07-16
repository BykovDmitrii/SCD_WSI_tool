#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
date
min_df=$7
echo $@
for cl in 3 ; do
for sil in True ; do
python clustering_evaluate.py \
        evaluate \
        --data_name=$1 \
        --output_directory=$2 \
        --subst1_path=$3 \
        --subst2_path=$4 \
        --subdir=modelNone \
        --use_silhouette=$sil \
        --k=10 \
        --n=15 \
        --lemmatizing_method=$5 \
        --vectorizer_name=$6 \
        --max_df=0.8 \
	--min_df=0.03 \
        --topk=150 \
        --max_number_clusters=$cl \
	--should_dump_results=True 
done
done
date
