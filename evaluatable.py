from abc import ABC, abstractmethod
import os
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from xlm.wsi import load_target_words
import time
import pandas as pd
from pathlib import Path

golden_data_path = str(Path(__file__).parent.absolute()) + '/data/golden_data'
def get_golden_data_paths(data_name):
    tmp_path1 = golden_data_path + '/task1'
    tmp_path2 = golden_data_path + '/task2'

    filename = data_name
    if 'rumacro' in data_name:
        filename = 'rumacro'

    paths1 = [tmp_path1 + '/' + i for i in os.listdir(tmp_path1) if filename in i]
    paths2 = [tmp_path2 + '/' + i for i in os.listdir(tmp_path2) if filename in i]

    if len(paths1) == 0 or len(paths2) == 0:
        print("no golden data for data %s" % data_name)
        return None, None

    assert len(paths1) == 1 and len(paths2) == 1, "invalid data name %s" % data_name
    return paths1[0], paths2[0]

class Evaluatable(ABC):
    def __init__(self, should_dump = False):
        self.output_dir = './'
        self.should_dump = should_dump

    @abstractmethod
    def solve(self, target_words):
        """
        main method.
        returns tuples of (word, ranking_metric, binary_label)
        """
        pass

    @abstractmethod
    def get_params(self : dict):
        """
        expected to return dict of parameters names and values
        """
        pass

    def load_golden_data_file(self, filename):
        with open(filename) as input:
            lines = input.readlines()

        res = dict()
        splitter = ','
        if splitter not in lines[0] and '\t' in lines[0]:
            splitter = '\t'

        for line in lines:
            parts = line.strip().split(splitter)
            try:
                res[parts[0].split('_')[0]] = float(parts[1])
            except:
                print("wrong string in target file - %s" % line)
        return res

    def load_golden_data(self, data_name):
        golden_binary_path, golden_ranking_path = get_golden_data_paths(data_name)
        if golden_binary_path is None or golden_ranking_path is None:
            return None, None
        ranking_data = self.load_golden_data_file(golden_ranking_path)
        binary_data = self.load_golden_data_file(golden_binary_path)
        binary_data = dict([(i, int(j)) for i,j in binary_data.items()])
        return ranking_data, binary_data

    def get_ranking_metric(self, ranking, ranking_gold):
        keys = sorted(ranking.keys())
        if keys != sorted(ranking_gold.keys()):
            print("Error - different ranking target words!")
            print(len(keys), keys, '\n')
            print(len(ranking_gold), (ranking_gold.keys()))

        values_1 = [ranking[k] for k in keys]
        values_2 = [ranking_gold[k] for k in keys]
        rho, p = spearmanr(values_1, values_2, nan_policy='raise')
        return rho

    def get_binary_metrics(self, bin_res, bin_res_gold):
        keys = sorted(bin_res.keys())
        if keys != sorted(bin_res.keys()):
            print("Error - different binary target words!")
            print(len(keys), keys, '\n')
            print(len(bin_res_gold), (bin_res_gold.keys()))
        values = [bin_res[k] for k in keys]
        values_gold = [bin_res_gold[k] for k in keys]
        return accuracy_score(values, values_gold), f1_score(values, values_gold, average='macro'), \
               dict(zip(keys, zip(values, values_gold)))

    def get_dataframe(self, ranking_metric, binary_accuracy_metric, binary_f1_metric):
        """
        returns dataframe containing calculated metrics and passed parameters of the algorithm
        """
        params = self.get_params()
        df_dict = params.copy()
        df_dict['ranking_score'] = ranking_metric
        df_dict['binary_accuracy_score'] = binary_accuracy_metric
        df_dict['binary_f1_score'] = binary_f1_metric
        df = pd.DataFrame.from_dict(df_dict)
        return df

    def evaluate(self, should_dump_results=True):
        """
        loads golden data, runs the solutions and computes metrics
        dumps the per-word results as text and metrics (with params) as dataframe
        returns that dataframe
        accaptable data_name:
            rumacro
            english
            swedish
            latin
            german
        """
        golden_ranking_data, golden_binary_data = self.load_golden_data(self.data_name)

        assert golden_binary_data is not None and golden_ranking_data is not None, "No golden data for %s" % self.data_name

        target_words = load_target_words(self.data_name)
        target_words = [i.split('_')[0] for i in target_words]
        results = self.solve(target_words)

        ranking_results = dict([(i[0], i[1]) for i in results])
        binary_results = dict([(i[0], i[2]) for i in results])

        ranking_metric= self.get_ranking_metric(ranking_results, golden_ranking_data)
        binary_accuracy_metric, binary_f1_metric, labels_pairs = self.get_binary_metrics(binary_results, golden_binary_data)

        line = 'ranking spearman rho = %f; binary accuracy = %f, macro f1_measure = %f' %\
                    (ranking_metric, binary_accuracy_metric, binary_f1_metric)
        print(line)

        dataframe = self.get_dataframe(ranking_metric, binary_accuracy_metric, binary_f1_metric)

        if should_dump_results:
            self.dump_results(results, dataframe)

        return dataframe, labels_pairs


    # results_format = list[(word, score, binary_label)]
    def dump_results(self, results, dataframe = None):
        """
        writing results to the files
        results are stored in <output_dir>/<template>/
        """
        params = self.get_params()
        params['timestamp'] = round(time.time() * 1000)

        output_dir = self.output_dir + '/' + params['template']
        os.makedirs(output_dir, exist_ok=True)

        filename_rank = '_'.join([str(v) for k, v in params.items() if k != 'template']) + '_rank.txt'
        filename_binary = '_'.join([str(v) for k, v in params.items() if k != 'template']) + '_binary.txt'
        filename_df = '_'.join([str(v) for k, v in params.items() if k != 'template']) + '_df.csv'

        full_path_rank = output_dir + '/' + filename_rank
        full_path_binary = output_dir + '/' + filename_binary
        full_path_df = output_dir + '/' + filename_df

        if results is not None:
            with open(full_path_rank, 'w+') as output_rank:
                with open(full_path_binary, 'w+') as output_binary:
                    for word, score, binary_label in results:
                        line_rank = word + '\t' + str(score) + '\n'
                        output_rank.write(line_rank)

                        line_binary = word + '\t' + str(binary_label) + '\n'
                        output_binary.write(line_binary)
        if dataframe is not None:
            dataframe.to_csv(full_path_df, sep='\t')

    def run(self, target_words, df1, df2):
        """
        runs solver and dumps the results without calculatin metrics
        """
        results = self.solve(target_words, df1, df2)
        self.dump_results(results)


class GridSearch(ABC):
    def __init__(self):
        self.stream = None

    @abstractmethod
    def get_params_list(self):
        """
        returns list of dicts {<param_name> : <param_value>} of all searchable parameters
        list should contain all variants of the parameters values to search through
        """
        pass

    @abstractmethod
    def create_evaluatable(self, data_name, params):
        """
        creates Evaluatable object with passed params
        "params" is a dict {<param_name> : <param_value>}
        """
        pass

    @abstractmethod
    def get_output_path(self):
        """
        return the full path to the output file (but no extension is required)
        the path must be valid, all directories on the path must exist
        """
        pass

    def dump_df(self, df):
        """
        just dumping dataframe
        """
        output_filename = self.get_output_path() + str(round(time.time() * 1000)) + '.csv'
        df.to_csv(output_filename, sep='\t')

    def test(self, data_name):
        params_list = self.get_params_list()
        for p in params_list:
            print(p)

    def search(self, data_name):
        """ run gridsearch. Params:
        data_name - name of the data to run gridsearch on. For short version of rumacro dataset use rumacroold
        """
        params_list = self.get_params_list()
        dfs = []
        for params in params_list:
            print("evaluating params ", list(params.items()))
            evaluatable = self.create_evaluatable(data_name, params)
            if evaluatable is None:
                 print("warining! evaluatable is None for params ", params)
                 continue
            df, _ = evaluatable.evaluate()
            dfs.append(df)
        big_df = pd.concat(dfs, ignore_index=True)
        self.dump_df(big_df)
        return self.stream

