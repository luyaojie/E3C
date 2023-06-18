#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019/10/20

from collections import Counter, OrderedDict, defaultdict
from typing import Tuple

import numpy as np
from allennlp.training.metrics.metric import Metric
from overrides import overrides
from scipy.optimize import linear_sum_assignment

DEBUG = False

if DEBUG:
    gold_output = open('golden.conll', 'w')
    pred_output = open('predict.conll', 'w')
    document_span_output = open('pred.document.soan', 'w')


def to_span_label_dict(top_spans, labels):
    """

    :param top_spans: ``torch.Tensor`` (Top Span, 2)
    :param labels: ``torch.Tensor`` (Top Span, 1)
    :return:
        (start, end) -> label
    """
    span_label_dict = dict()
    for span, label in zip(top_spans.tolist(), labels.tolist()):
        span_label_dict[tuple(span)] = label
    return span_label_dict


def get_mention_to_cluster_id_dict(mention_to_clusters):
    cluster_id_dict = dict()
    for cluster in mention_to_clusters.values():
        if cluster not in cluster_id_dict:
            cluster_id_dict[cluster] = len(cluster_id_dict)

    mention_to_cluster_id_dict = dict()
    for mention in mention_to_clusters:
        mention_to_cluster_id_dict[mention] = cluster_id_dict[mention_to_clusters[mention]]

    return mention_to_cluster_id_dict


def to_conll(mentions_clusters_list, tokens, doc_id):
    output = list()
    output += ["#begin document (%s); part 000" % doc_id]
    for mention_index, (mention, cluster_id) in enumerate(mentions_clusters_list):
        if mention is None:
            output += ["%s\t%s\t%s\t(%s)" % (doc_id,
                                             mention_index,
                                             "None",
                                             '-')]
        else:
            output += ["%s\t%s\t%s\t(%s)" % (doc_id,
                                             mention_index,
                                             "_".join(tokens[mention[0]: mention[1] + 1]),
                                             cluster_id)]
    output += ["#end document"]
    return '\n'.join(output)


def mapping_predict_golden(mention_to_gold, mention_to_pred):
    mention_to_gold_list = [(key, mention_to_gold[key]) for key in mention_to_gold.keys()]
    mention_to_pred_list = []
    for gm in mention_to_gold:
        if gm not in mention_to_pred:
            mention_to_gold_list += [(None, '-')]

    for gm, _ in mention_to_gold_list:
        if gm in mention_to_pred:
            mention_to_pred_list += [(gm, mention_to_pred[gm])]
        else:
            mention_to_pred_list += [(None, '-')]

    for pm in mention_to_pred:
        if pm not in mention_to_gold:
            mention_to_pred_list += [(pm, mention_to_pred[pm])]
    return mention_to_gold_list, mention_to_pred_list


def output_mapping_conll(mention_to_gold, mention_to_predicted, metadata):
    get_ordered_dict = lambda x: OrderedDict({key: x[key] for key in sorted(x.keys())})
    mention_to_gold = get_ordered_dict(mention_to_gold)
    mention_to_predicted = get_ordered_dict(mention_to_predicted)

    document_span_output.write('%s\t%s\n' % (metadata['doc_id'], len(mention_to_predicted)))
    result = mapping_predict_golden(get_mention_to_cluster_id_dict(mention_to_gold),
                                    get_mention_to_cluster_id_dict(mention_to_predicted))
    mention_to_gold_list, mention_to_pred_list = result

    predict_list = to_conll(mention_to_pred_list, metadata['original_text'],
                            metadata['doc_id'])
    golden_list = to_conll(mention_to_gold_list, metadata['original_text'],
                           metadata['doc_id'])
    pred_output.write(predict_list + '\n')
    gold_output.write(golden_list + '\n')


def clusters_to_mention_to_predicted(clusters):
    mention_to_predicted = dict()
    for cluster in clusters:
        for mention in cluster:
            mention_to_predicted[mention] = cluster
    return mention_to_predicted


def span_label_list_to_dict(span_list):
    span_dict = dict()
    for span in span_list:
        span_dict[(span[0], span[1])] = span[2]
    return span_dict


def combine_cluster_with_label(span_list, clusters):
    new_clusters = list()
    predicted_dict = span_label_list_to_dict(span_list)
    for cluster in clusters:
        new_cluster = list()
        for mention in cluster:
            assert len(mention) == 2
            new_cluster += [(mention[0], mention[1], predicted_dict[(mention[0], mention[1])])]
        new_clusters += [tuple(new_cluster)]
    return new_clusters


def split_cluster_with_different_label(clusters):
    new_clusters = list()
    for cluster in clusters:
        new_label_cluster = defaultdict(list)
        for mention in cluster:
            assert len(mention) == 3
            new_label_cluster[mention[2]] += [mention]
        for label, new_cluster in new_label_cluster.items():
            new_clusters += [tuple(new_cluster)]
    return new_clusters


@Metric.register("event_coref_scores")
class EventCorefScores(Metric):
    def __init__(self, mapping_type=False) -> None:
        self.scorers = [Scorer(m) for m in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]
        self._mapping_type = mapping_type

    @overrides
    def __call__(self,  # type: ignore
                 predicted_clusters_list,
                 metadata_list,
                 pred_label_spans=None,
                 gold_label_spans=None,
                 ):
        """
        Parameters
        ----------
        predicted_clusters_list : ``List[List[Tuple[Tuple[Int, Int], Tuple[Int, Int], ...]]]``

            [((start11, end11), (start12, end12), (start13, end13), ),
             ((start21, end21), (start22, end22), ),
            ]
        """
        for i, metadata in enumerate(metadata_list):
            predicted_clusters = predicted_clusters_list[i]
            mention_to_predicted = clusters_to_mention_to_predicted(predicted_clusters)
            gold_clusters, mention_to_gold = self.get_gold_clusters(metadata["clusters"])

            if self._mapping_type:
                assert pred_label_spans is not None
                assert gold_label_spans is not None

                predicted_clusters = combine_cluster_with_label(pred_label_spans[i], predicted_clusters)
                gold_clusters = combine_cluster_with_label(gold_label_spans[i], gold_clusters)

                predicted_clusters = split_cluster_with_different_label(predicted_clusters)
                gold_clusters = split_cluster_with_different_label(gold_clusters)

                mention_to_predicted = clusters_to_mention_to_predicted(predicted_clusters)
                mention_to_gold = clusters_to_mention_to_predicted(gold_clusters)

            if DEBUG:
                output_mapping_conll(mention_to_predicted=mention_to_predicted,
                                     mention_to_gold=mention_to_gold,
                                     metadata=metadata)

            for scorer in self.scorers:
                scorer.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        metrics = (lambda e: e.get_precision(), lambda e: e.get_recall(), lambda e: e.get_f1())
        precision, recall, f1_score = tuple(sum(metric(e) for e in self.scorers) / len(self.scorers)
                                            for metric in metrics)
        if DEBUG:
            for scorer in self.scorers:
                print(scorer, scorer.get_f1())
        if reset:
            self.reset()
        return precision, recall, f1_score

    @overrides
    def reset(self):
        self.scorers = [Scorer(metric) for metric in (Scorer.muc, Scorer.b_cubed, Scorer.ceafe)]

    @staticmethod
    def get_gold_clusters(gold_clusters):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gold_cluster in gold_clusters:
            for mention in gold_cluster:
                mention_to_gold[mention] = gold_cluster
        return gold_clusters, mention_to_gold


class Scorer:
    """
    Mostly borrowed from <https://github.com/clarkkev/deep-coref/blob/master/evaluation.py>
    TAC 2017 Event also evaluate cluster size is 1.
    """

    def __init__(self, metric):
        self.precision_numerator = 0
        self.precision_denominator = 0
        self.recall_numerator = 0
        self.recall_denominator = 0
        self.metric = metric

    def __repr__(self):
        return str(self.metric)

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == self.ceafe:  # pylint: disable=comparison-with-callable
            p_num, p_den, r_num, r_den = self.metric(predicted, gold)
        else:
            p_num, p_den = self.metric(predicted, mention_to_gold)
            r_num, r_den = self.metric(gold, mention_to_predicted)
        self.precision_numerator += p_num
        self.precision_denominator += p_den
        self.recall_numerator += r_num
        self.recall_denominator += r_den

    def get_f1(self):
        precision = 0 if self.precision_denominator == 0 else \
            self.precision_numerator / float(self.precision_denominator)
        recall = 0 if self.recall_denominator == 0 else \
            self.recall_numerator / float(self.recall_denominator)
        return 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def get_recall(self):
        if self.recall_numerator == 0:
            return 0
        else:
            return self.recall_numerator / float(self.recall_denominator)

    def get_precision(self):
        if self.precision_numerator == 0:
            return 0
        else:
            return self.precision_numerator / float(self.precision_denominator)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    @staticmethod
    def b_cubed(clusters, mention_to_gold):
        """
        Averaged per-mention precision and recall.
        <https://pdfs.semanticscholar.org/cfe3/c24695f1c14b78a5b8e95bcbd1c666140fd1.pdf>
        """
        numerator, denominator = 0, 0
        for cluster in clusters:
            # TAC 2017 Event also evaluate cluster size is 1.
            # if len(cluster) == 1:
            #     continue
            gold_counts = Counter()
            correct = 0
            for mention in cluster:
                if mention in mention_to_gold:
                    gold_counts[tuple(mention_to_gold[mention])] += 1
            for cluster2, count in gold_counts.items():
                # if len(cluster2) != 1:
                correct += count * count
            numerator += correct / float(len(cluster))
            denominator += len(cluster)
        return numerator, denominator

    @staticmethod
    def muc(clusters, mention_to_gold):
        """
        Counts the mentions in each predicted cluster which need to be re-allocated in
        order for each predicted cluster to be contained by the respective gold cluster.
        <https://aclweb.org/anthology/M/M95/M95-1005.pdf>
        """
        true_p, all_p = 0, 0
        for cluster in clusters:
            all_p += len(cluster) - 1
            true_p += len(cluster)
            linked = set()
            for mention in cluster:
                if mention in mention_to_gold:
                    linked.add(mention_to_gold[mention])
                else:
                    true_p -= 1
            true_p -= len(linked)
        return true_p, all_p

    @staticmethod
    def phi4(gold_clustering, predicted_clustering):
        """
        Subroutine for ceafe. Computes the mention F measure between gold and
        predicted mentions in a cluster.
        """
        return 2 * len([mention for mention in gold_clustering if mention in predicted_clustering]) \
               / float(len(gold_clustering) + len(predicted_clustering))

    @staticmethod
    def ceafe(clusters, gold_clusters):
        """
        Computes the  Constrained EntityAlignment F-Measure (CEAF) for evaluating coreference.
        Gold and predicted mentions are aligned into clusterings which maximise a metric - in
        this case, the F measure between gold and predicted clusters.

        <https://www.semanticscholar.org/paper/On-Coreference-Resolution-Performance-Metrics-Luo/de133c1f22d0dfe12539e25dda70f28672459b99>
        """
        # TAC 2017 Event also evaluate cluster size is 1.
        # clusters = [cluster for cluster in clusters if len(cluster) != 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = Scorer.phi4(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])
        return similarity, len(clusters), similarity, len(gold_clusters)
