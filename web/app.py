import os
from clustering_evaluate import Clustering_Search, Clustering_Pipeline
import io
import fire
from xlm.data_loading import load_target_words
from xlm.wsi_evaluation import evaluate as evaluate_wsi

from flask import (
    Flask, Blueprint, flash, g, redirect, render_template, request, session, url_for, send_file
)
from werkzeug.security import check_password_hash, generate_password_hash

# from flaskr.db import get_db

def run_wsi(evaluatable, data_name):
    target_words = load_target_words(data_name)
    if target_words is None:
        target_words = evaluatable.subst1['word'].unique()
    else:
        target_words = [i.split('_')[0] for i in target_words]
    evaluatable.solve(target_words, data_name + '_1', None, data_name + '_2', None)
    df = evaluatable.subst1
    return evaluate_wsi(df=df, stream=evaluatable.stream)

def create_app(evaluatable, data_name):

    def get_words_and_labels():
        for key in label_pairs:
            yield key, label_pairs[key]

    def get_wsi_words_and_ari():
        for key in wsi_words:
            print(wsi_words[key])
            yield key, wsi_words[key]

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    app.jinja_env.globals.update(get_words_and_labels=get_words_and_labels)
    app.jinja_env.globals.update(get_wsi_words_and_ari=get_wsi_words_and_ari)

    print("search created")
    evaluatable._prepare(data_name + '_1', None, data_name + '_2', None)
    ranking_golden, binary_golden = evaluatable.load_golden_data(data_name)
    mode = 'scd'
    if ranking_golden is None or binary_golden is None:
        mode = 'wsi'
        wsi_words = run_wsi(evaluatable, data_name)
    else:
        _, label_pairs = evaluatable.evaluate(should_dump_results=False)

    pictures_dir = './pictures'
    os.makedirs(pictures_dir, exist_ok=True)

    @app.route('/')
    def words():
        return render_template('words.html', mode=mode)
    # a simple page that says hello

    @app.route('/get_word_plot/<path>')
    def get_word_plot(path):
        print(path)
        return send_file(path.replace("_", "/"), mimetype='image/jpg', cache_timeout=0)

    @app.route('/get_dist_hist/<path>')
    def get_dist_hist(path):
        print(path)
        return send_file(path.replace("_", "/"), mimetype='image/jpg', cache_timeout=0)

    @app.route('/analyze/<word>')
    def analyze(word):
        if mode == 'scd':
            stream = io.StringIO()
            binary, distance = evaluatable.solve_for_one_word(word, stream)
            label_pairs = {word:(binary, binary_golden[word])}
            wp_path, dh_path, df = evaluatable.analyze_error(word, stream, label_pairs, pictures_dir)

            return render_template('analysis.html', text=stream.getvalue(), wp_path=wp_path, dh_path=dh_path, word=word, df=df, mode=mode)
        elif mode == 'wsi':
            stream = io.StringIO()
            binary, distance = evaluatable.solve_for_one_word(word, stream)
            wp_path, dh_path, df = evaluatable.analyze_error(word, stream, None, pictures_dir)
            return render_template('analysis.html', text=stream.getvalue(), wp_path=wp_path, dh_path=dh_path, word=word, df=df, mode=mode)

    return app

def configure_and_run(data_name, output_directory,subst1_path, subst2_path = None, vectorizer_name = 'count', min_df = 0.03, max_df = 0.8,
                      max_number_clusters = 0,
                 use_silhouette = True, k = 10, n = 15, topk = 150, lemmatizing_method = 'single', binary = False,
                     drop_duplicates=False, count_lemmas_weights=True):

    if subst2_path is None:
        subst2_path = subst1_path

    evaluatable = Clustering_Pipeline(data_name, output_directory, vectorizer_name=vectorizer_name, min_df=min_df, max_df=max_df,
                                      max_number_clusters=max_number_clusters, use_silhouette=use_silhouette, k=k, n=n, topk=topk,
                                      lemmatizing_method=lemmatizing_method, drop_duplicates=drop_duplicates,
                                      count_lemmas_weights=count_lemmas_weights,
                                      binary=binary,dump_errors=True, path_1=subst1_path, path_2=subst2_path)
    app = create_app(evaluatable, data_name)
    app.run()

if __name__ == '__main__':
    fire.Fire(configure_and_run)

#TODO: добавить delete words parts