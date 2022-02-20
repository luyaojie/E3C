#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import sys
from collections import Counter
import argparse
import tabulate

from src.data.utils import kbp_label_set

keys = [
    'doc', 'sent', 'token',
    'mention', 'event',  # 'SingleMention', 'MultiMention',
    'mention*', 'event*',  # 'SingleMention*', 'MultiMention*',
]
show_keys = [
    'Documents', 'Sentences', 'Tokens',
    'Event mentions', 'Event chains',  # 'Single Mention', 'Multiple Mention',
    'Event mentions*', 'Event chains*',  # 'Single Mention*', 'Multiple Mention*'
]


def format_number(value):
    return f'{value:,}'


def count_one_file(in_filename, file_type):
    counter = Counter()
    with open(in_filename) as fin:
        for line in fin:
            document = json.loads(line)
            counter.update(['doc'])
            for sentence in document['sentences']:
                counter.update(['sent'])
                for token in sentence['tokens']:
                    counter.update(['token'])

            for event in document['event']:
                event_type = event['mentions'][0]['type'] + ':' + event['mentions'][0]['subtype']

                counter.update(['event*'])
                for mention in event['mentions']:
                    counter.update(['mention*'])

                # if len(event['mentions']) == 1:
                #     counter.update(['SingleMention*'])
                # else:
                #     counter.update(['MultiMention*'])

                if event_type in kbp_label_set:

                    counter.update(['event'])
                    for mention in event['mentions']:
                        counter.update(['mention'])

                    # if len(event['mentions']) == 1:
                    #     counter.update(['SingleMention'])
                    # else:
                    #     counter.update(['MultiMention'])

        for key in keys:
            value = counter[key]
            # print("%s, %s: %s" % (file_type, key, value))
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', dest='data')
    options = parser.parse_args()

    global_counter = Counter()
    data_table_dict = dict()

    if os.path.isdir(options.data):

        for file_type in ['train', 'valid', 'test']:
            in_filename = options.data + os.sep + file_type + '.jsonl'
            counter = count_one_file(in_filename, file_type)
            data_table_dict[file_type] = counter
            global_counter.update(counter)

        row_keys = ['train', 'valid', 'test', 'total']

        data_table = list()
        data_table_dict['total'] = global_counter
        for key, show_key in zip(keys, show_keys):
            data_row = [show_key] + [format_number(data_table_dict[row][key]) for row in row_keys]
            data_table += [data_row]

        print(tabulate.tabulate(data_table, headers=[] + row_keys, tablefmt='latex', stralign='right'), )
    else:
        in_filename = options.data
        print(in_filename)
        count_one_file(in_filename, "File")


if __name__ == "__main__":
    main()
