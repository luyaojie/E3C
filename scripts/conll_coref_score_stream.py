#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019/10/26
import sys

separator = "\t"


def pprint(scores_by_metric):
    output_list = []
    for key in ['filename', 'mention_type_p', 'mention_type_r', 'mention_type', 'bcub', 'ceafe', 'muc', 'blanc',
                'AVG-F']:
        output_list += [scores_by_metric.get(key, '00.00')]
    print(separator.join(output_list))


def get_conll_scores():
    scores_by_metric = {}
    print(separator.join(['filename', 'mention_type_p', 'mention_type_r', 'mention_type', 'bcub', 'ceafe', 'muc', 'blanc', 'AVG-F']))
    with sys.stdin as f:
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
            # elif line.startswith('[WARNING]'):
            #     print(line)
            if line.endswith('Evaluation Done.'):
                pprint(scores_by_metric)
                scores_by_metric = {}
    return scores_by_metric


if __name__ == "__main__":
    get_conll_scores()
