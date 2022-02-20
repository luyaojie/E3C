#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os
import re
from collections import OrderedDict, defaultdict

import numpy
import numpy as np
from tabulate import tabulate

separator = "\t"

expected_file_list = ['valid_notyped',
                      'test_notyped',
                      'valid_typed',
                      'test_typed'
                      ]


def get_conll_scores(filename):
    if not os.path.exists(filename):
        return {}
    scores_by_metric = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("==>"):
                filename = line.replace('==>', '').replace('<==', '').strip()
                scores_by_metric['filename'] = filename
            elif line.startswith("Metric") and not line.endswith('*'):
                att = line.split()
                name = att[2]
                f1 = att[-1]
                scores_by_metric[name] = f1
            elif line.startswith('Overall Average CoNLL score'):
                name = "AVG-F"
                f1 = line.split()[-1]
                scores_by_metric[name] = f1
            elif line.startswith('mention_type\t'):
                scores_by_metric['mention_type'] = line.split()[3]
                scores_by_metric['mention_type_r'] = line.split()[2]
                scores_by_metric['mention_type_p'] = line.split()[1]
            elif line.startswith('plain\t'):
                scores_by_metric['plain'] = line.split()[3]
                scores_by_metric['plain_r'] = line.split()[2]
                scores_by_metric['plain_p'] = line.split()[1]
            elif line.startswith('[WARNING]'):
                print(line)
    for key in scores_by_metric:
        if not isinstance(scores_by_metric[key], float):
            scores_by_metric[key] = float(scores_by_metric[key])
    return scores_by_metric


def overall_run_result(final_result_summary):
    result_dict = defaultdict(list)
    mean_result_dict = dict()
    max_result_dict = dict()

    for result in final_result_summary:
        name = re.subn(r"_run\d+", "", result[0])[0]
        result_dict[name] += [result[1:]]

    for name in result_dict:
        # run_time = '-%d-Run' % len(result_dict[name])
        run_time = ''
        mean_result_dict[name + run_time] = np.array(result_dict[name]).mean(0)
        max_result_dict[name + run_time] = np.array(result_dict[name]).max(0)

    return {'mean': mean_result_dict, 'max': max_result_dict}


def dict_to_table(overall_result):
    data_table = list()
    for name, value in overall_result.items():
        row = [name] + value.tolist()
        data_table += [row]
    return data_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', nargs='+', required=True, dest='model_list', type=str)
    parser.add_argument('-f', dest='table_format', default='grid')
    parser.add_argument('-mean', dest='mean', action='store_true')
    parser.add_argument('-max', dest='max', action='store_true')
    parser.set_defaults(ace=True)
    option = parser.parse_args()

    final_result_summary = defaultdict(list)
    data_table = list()
    header = ['model_path']
    for fn in expected_file_list:
        header += [fn + 'Type-F', fn + 'AVG-']

    for model_folder in option.model_list:
        print(model_folder)
        result = OrderedDict()
        for filename_suffix in expected_file_list:
            filename = model_folder + os.sep + filename_suffix + 'eval.result'
            if not os.path.exists(filename):
                filename = model_folder + os.sep + filename_suffix + '-eval.result'
            scores = get_conll_scores(filename)
            result[filename_suffix] = scores
            final_result_summary[filename_suffix] += [scores.get('AVG-F', 0.)]
        row = [model_folder]
        for fn in result:
            row += [result[fn].get('mention_type', 0.), result[fn].get('AVG-F', 0.)]
        data_table += [row]

    mean_row = ['mean'] + ["%.2f" % numpy.mean(final_result_summary[fn]) for fn in expected_file_list]
    max_row = ['max'] + ["%.2f" % numpy.max(final_result_summary[fn]) for fn in expected_file_list]
    print(tabulate(data_table + [mean_row] + [max_row], headers=header, tablefmt=option.table_format))

    if option.mean or option.max:
        overall_result = overall_run_result(data_table[1:])

        if option.mean:
            mean_data_table = dict_to_table(overall_result['mean'])
            print('Mean over Run')
            print(tabulate(mean_data_table, headers=header, tablefmt=option.table_format))

        if option.max:
            mean_data_table = dict_to_table(overall_result['max'])
            print('Max  over Run')
            print(tabulate(mean_data_table, headers=header, tablefmt=option.table_format))


if __name__ == "__main__":
    main()
