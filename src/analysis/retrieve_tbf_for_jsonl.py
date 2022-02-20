#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019/12/3
import argparse
import codecs
import json

from src.utils.tbf_utils import load_document_dict_from_tbf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='jsonl_path', required=True)
    parser.add_argument('-t', dest='tbf_path', required=True)
    parser.add_argument('-o', dest='output', required=True)
    args = parser.parse_args()

    gold_document_dict = load_document_dict_from_tbf(args.tbf_path)
    new_tbf_list = list()
    with codecs.open(args.jsonl_path, 'r', 'utf8') as fin:
        for doc_json in fin:
            document = json.loads(doc_json)
            doc_id = document['id']
            new_tbf_list += [gold_document_dict[doc_id].to_lines()]

    with codecs.open(args.output, 'w', 'utf8') as output:
        output.write('\n'.join(new_tbf_list))


if __name__ == "__main__":
    main()
