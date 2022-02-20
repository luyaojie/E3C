#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os


def main():
    file_type_list = ['train', 'valid', 'test']
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input_folder', type=str)
    parser.add_argument('-o', dest='output_folder', type=str)
    options = parser.parse_args()

    if not os.path.exists(options.output_folder):
        os.makedirs(options.output_folder)

    for file_type in file_type_list:
        filename = options.input_folder + os.sep + file_type + '.jsonl'
        file_list = [json.loads(line)['id'] for line in open(filename)]
        with open(options.output_folder + os.sep + file_type + '.filelist', 'w') as output:
            for line in file_list:
                output.write(line + '\n')


if __name__ == "__main__":
    main()
