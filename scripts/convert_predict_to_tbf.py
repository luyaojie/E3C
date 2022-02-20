#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019/9/25
import json
import sys

from allennlp.common import JsonDict

escape_set = {'NIL'}


class Span:
    def __init__(self, top_indices, type_label, realis_label):
        self._top_indices = top_indices
        self._type_label = type_label
        self._realis_label = realis_label


class Document:
    def __init__(self, doc_dict: JsonDict, system='System'):
        self._offset_list = doc_dict['offset']
        self._top_spans = doc_dict['top_spans']
        self._top_type_list = doc_dict['top_type_labels']
        self._top_realis_list = doc_dict.get('top_realis_labels', ['Actual'] * len(doc_dict['top_type_labels']))
        self._doc_id = doc_dict['doc_id']
        self._clusters = doc_dict.get('clusters', [])
        self._token_list = doc_dict['document']
        self._span_index_dict = self.init_span_index_dict(doc_dict)
        self._system = system
        assert len(self._token_list) == len(self._offset_list)

    def init_span_index_dict(self, doc_dict: JsonDict):
        _span_index_dict = dict()
        for _span in doc_dict['top_spans']:
            _span_index_dict[self.get_span_offset(_span)] = len(_span_index_dict)
        return _span_index_dict

    def get_span_offset(self, span):
        return "%s,%s" % (self._offset_list[span[0]][0], self._offset_list[span[-1]][1])

    def get_span_type(self, span):
        return "%s" % self._top_type_list[self._span_index_dict[self.get_span_offset(span)]]

    def get_span_realis(self, span):
        return "%s" % self._top_realis_list[self._span_index_dict[self.get_span_offset(span)]]

    def get_span_str(self, span):
        return ' '.join(self._token_list[span[0]:span[-1] + 1])

    def iter_span(self):
        for span in self._top_spans:
            yield span

    def iter_cluster(self):
        for cluster in self._clusters:
            yield cluster

    def convert_span(self, span):
        offset = self.get_span_offset(span)
        token_str = self.get_span_str(span)
        span_type = self.get_span_type(span)
        span_realis = self.get_span_realis(span)
        return '\t'.join([self._system,
                          self._doc_id,
                          'E%s' % self._span_index_dict[offset],
                          offset,
                          token_str,
                          span_type,
                          "Actual" if span_realis in escape_set else span_realis])

    def convert_cluster(self, cluster, cluster_index):
        return '\t'.join(['@Coreference',
                          'C%s' % cluster_index,
                          ','.join(['E%s' % self._span_index_dict[self.get_span_offset(span)] for span in cluster])])


def convert_json_to_tbf(doc_dict: JsonDict):
    document = Document(doc_dict, 'system')
    tb_list = list()
    doc_id = doc_dict['doc_id']
    tb_list += ["#BeginOfDocument %s" % doc_id]
    for span in document.iter_span():
        if document.get_span_type(span) in escape_set:
            continue
        tb_list += [document.convert_span(span)]
    for cluster_index, cluster in enumerate(document.iter_cluster()):
        if len(cluster) == 1:
            continue
        tb_list += [document.convert_cluster(cluster, cluster_index)]
    tb_list += ["#EndOfDocument"]
    return tb_list


def main():
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    output = open(output_filename, 'w')
    with open(input_filename) as fin:
        for line in fin:
            tb_list = convert_json_to_tbf(json.loads(line))
            output.write('\n'.join(tb_list))
            output.write('\n')


if __name__ == "__main__":
    main()
