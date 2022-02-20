#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2019/11/4
import os
from abc import abstractmethod
from typing import Tuple, Dict

import torch


def prepare_cluster():
    clusters = list()
    spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
    return clusters, spans_to_cluster_ids


def list_partition(numbers, value):
    left_numbers, right_numbers = list(), list()
    for number in numbers:
        if number <= value:
            left_numbers += [number]
        else:
            right_numbers += [number]
    return left_numbers, right_numbers


def add_mention_to_cluster(mention_span: Tuple[int, int], spans_to_cluster_ids, clusters, cluster_id: int = None):
    """
    Add Mention to Cluster
    :param mention_span: (start, end)
    :param spans_to_cluster_ids: (start, end) -> cluster id
    :param clusters: cluster id -> {(start1, end1), (start2, end2)}
    :param cluster_id:
    :return:
    """
    if cluster_id is not None:
        clusters[cluster_id].add(mention_span)
        spans_to_cluster_ids[mention_span] = cluster_id
        return cluster_id

    # Cluster ID is None, add antecedent
    if mention_span in spans_to_cluster_ids:
        predicted_cluster_id: int = spans_to_cluster_ids[mention_span]
    else:
        # We start a new cluster.
        predicted_cluster_id = len(clusters)
        # Append a new cluster containing only this span.
        clusters.append({mention_span})
        # Record the new id of this span.
        spans_to_cluster_ids[mention_span] = predicted_cluster_id
    return predicted_cluster_id


def change_list_clusters_to_tuple_clusters(clusters):
    for cluster_index in range(len(clusters)):
        clusters[cluster_index] = tuple(clusters[cluster_index])
    return clusters


def add_single_mention_as_cluster_to_clusters(pred_label_spans, spans_to_cluster_ids, clusters):
    """

    :param pred_label_spans: ``list``
        [(start1, end1, label1), (start2, end2, label2), ...]
    :param spans_to_cluster_ids:
    :param clusters:
    :return:
    """
    for span_start, span_end, span_label in pred_label_spans:
        span_key = (span_start, span_end)
        if span_key not in spans_to_cluster_ids:
            add_mention_to_cluster(span_key, spans_to_cluster_ids, clusters)
    return clusters


class ClusterDecoder:
    def __init__(self, antecedent_indices, vocab, positive_label_size=18):
        self._antecedent_indices = antecedent_indices
        self._vocab = vocab
        self._positive_label_size = positive_label_size

    def get_label_name(self, index):
        return self._vocab.get_token_from_index(index, 'labels')

    def get_antecedent_span(self, top_spans, span_index, coref_antecedent_index):
        antecedent_span_index = coref_antecedent_index - (self._positive_label_size + 1)
        coref_antecedent_span_index = self._antecedent_indices[span_index, antecedent_span_index]
        coref_antecedent_span = top_spans[coref_antecedent_span_index]
        return coref_antecedent_span[0].item(), coref_antecedent_span[1].item()

    @abstractmethod
    def decode(self, top_spans, coreference_scores):
        pass


class TypeGuidedClusterDecoder(ClusterDecoder):

    def __init__(self, antecedent_indices, vocab, positive_label_size=18):
        super().__init__(antecedent_indices, vocab, positive_label_size)

    def decode(self, top_spans, coreference_scores):
        clusters = list()
        spans_to_cluster_ids: Dict[Tuple[int, int], int] = dict()
        positive_top_spans_dict: Dict[Tuple[int, int], str] = dict()
        for span_index, (current_span, coreference_score) in enumerate(zip(top_spans, coreference_scores)):

            current_span = tuple(current_span.tolist())
            sorted_indices = torch.argsort(coreference_score, descending=True).tolist()

            # 0 is NIL Span, find antecedent index greater than 0
            positive_antecedent_indices = sorted_indices[:sorted_indices.index(0)]

            if len(positive_antecedent_indices) == 0:
                # NIL Span
                continue

            label_antecedent_indices, coref_antecedent_indices = list_partition(positive_antecedent_indices[:1],
                                                                                self._positive_label_size)
            if sorted_indices[0] in label_antecedent_indices:
                # Label Scores is bigger
                # this span links to label
                # Add span to Positive Span Dict
                current_span_label = self.get_label_name(label_antecedent_indices[0])
                positive_top_spans_dict[current_span] = current_span_label
            else:
                antecedent_index = coref_antecedent_indices[0]
                antecedent_span = self.get_antecedent_span(top_spans,
                                                           span_index,
                                                           antecedent_index)

                antecedent_span_label = positive_top_spans_dict.get(antecedent_span, None)

                if antecedent_span_label is None:
                    # print("Antecedent is None, skip this pair")
                    continue

                # this span's label is same to antecedent span's label
                positive_top_spans_dict[current_span] = antecedent_span_label

                predicted_cluster_id = add_mention_to_cluster(antecedent_span,
                                                              spans_to_cluster_ids,
                                                              clusters)

                add_mention_to_cluster(current_span,
                                       spans_to_cluster_ids,
                                       clusters,
                                       predicted_cluster_id)

        pred_label_spans = [(span_key[0], span_key[1], span_label)
                            for span_key, span_label in positive_top_spans_dict.items()]

        add_single_mention_as_cluster_to_clusters(pred_label_spans, spans_to_cluster_ids, clusters)

        change_list_clusters_to_tuple_clusters(clusters)

        positive_top_spans = list()
        positive_top_spans_labels = list()
        for span, span_label in positive_top_spans_dict.items():
            positive_top_spans += [span]
            positive_top_spans_labels += [span_label]

        return {"positive_top_spans": positive_top_spans,
                "positive_top_spans_labels": positive_top_spans_labels,
                "pred_label_spans": pred_label_spans,
                "clusters": clusters
                }


def node_decode(output_dict, vocab, positive_label_size=18, decoding_algorithm='type-guided', type_threshold=-1):
    # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
    # the start and end indices of each span.
    batch_top_spans = output_dict["top_spans"].detach().cpu()
    batch_coreference_scores = output_dict['coreference_scores'].detach().cpu()

    # A tensor of shape (num_spans_to_keep, max_antecedents), representing the indices
    # of the predicted antecedents with respect to the 2nd dimension of ``batch_top_spans``
    # for each antecedent we considered.
    antecedent_indices = output_dict["antecedent_indices"].detach().cpu()

    if decoding_algorithm == 'type-guided':
        cluster_decoder = TypeGuidedClusterDecoder(antecedent_indices, vocab, positive_label_size)
    else:
        raise NotImplementedError('Decoding Algorithm Not Implemented: %s' % decoding_algorithm)

    batch_positive_top_spans = list()
    batch_positive_top_spans_labels = list()
    batch_pred_label_spans = list()
    batch_clusters = list()

    for top_spans, coreference_scores in zip(batch_top_spans, batch_coreference_scores):
        result = cluster_decoder.decode(top_spans, coreference_scores)
        batch_positive_top_spans += [result['positive_top_spans']]
        batch_positive_top_spans_labels += [result['positive_top_spans_labels']]
        batch_pred_label_spans += [result['pred_label_spans']]
        batch_clusters += [result['clusters']]

    output_dict['top_spans'] = batch_positive_top_spans
    output_dict['top_type_labels'] = batch_positive_top_spans_labels
    output_dict['clusters'] = batch_clusters
    output_dict['pred_label_spans'] = batch_pred_label_spans
    return output_dict


if __name__ == "__main__":
    pass
