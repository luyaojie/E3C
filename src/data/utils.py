#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by Roger on 2019/10/29
import collections
from typing import DefaultDict, List, Tuple, Set

kbp_label_set = set("""conflict:attack
conflict:demonstrate
contact:broadcast
contact:contact
contact:correspondence
contact:meet
justice:arrestjail
life:die
life:injure
manufacture:artifact
movement:transportartifact
movement:transportperson
personnel:elect
personnel:endposition
personnel:startposition
transaction:transaction
transaction:transfermoney
transaction:transferownership""".split('\n'))


def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]


def get_safe_start_end(span_start, span_end, document_length, context_window):
    safe_start = max(0, span_start - context_window)
    safe_end = min(document_length, span_end + context_window)
    return safe_start, safe_end


def process_sentences(document, logger_counter=None, max_length=1024, label_set=None):
    """
    Processing Document Entity-Relation-Event to Sentences
    :param document:
    :param logger_counter ``Counter``
        Logger for Count Instances
    :param max_length ``int``
        Max Length
    :param label_set ``set``
        Valid Label Set
    :return:
        sentences ``List[List[Dict]]``
            A list of Sentence
                Sentence is a list of token dicts
        sentences_span_label_dict ``List[Dict]``
            A list of span_label_dict
                span_label_dict is a dict of token dicts
                (span_start, span_end) -> (event type label, event realis label)
    """
    sentences = list()
    sentences_span_label_dict = list()

    for sentence_index, sentence in enumerate(document['sentences']):

        token_to_index_map = dict()
        total_tokens = 0

        for token_index, token in enumerate(sentence['tokens']):
            if token_index > max_length:
                break
            token_to_index_map[(sentence_index, token_index)] = total_tokens
            total_tokens += 1

        sentences += [sentence['tokens']]

        # logger.info("event num: %s" % len(document['event']))
        span_label_dict = dict()
        for event_index, event in enumerate(document['event']):
            # logger.info("mention num: %s" % len(event['mentions']))
            for event_mention_index, event_mention in enumerate(event['mentions']):

                if len(event_mention['nugget']['tokens']) == 0:
                    logger_counter.update(['mention escape: token not found'])
                    continue
                if tuple(event_mention['nugget']['tokens'][0]) not in token_to_index_map:
                    break
                if tuple(event_mention['nugget']['tokens'][-1]) not in token_to_index_map:
                    break

                event_mention_label = "%s:%s" % (event_mention['type'], event_mention['subtype'])
                event_mention_realis = "%s" % event_mention['realis']

                span_start = token_to_index_map[tuple(event_mention['nugget']['tokens'][0])]
                span_end = token_to_index_map[tuple(event_mention['nugget']['tokens'][-1])]
                span_key = (span_start, span_end)

                if label_set is not None and event_mention_label not in label_set:
                    logger_counter.update(['mention escape: label'])
                    continue

                if span_key in span_label_dict:
                    logger_counter.update(['mention overlap'])
                    continue

                span_label_dict[span_key] = (event_mention_label, event_mention_realis)

                logger_counter.update(['mention keep'])

        sentences_span_label_dict += [span_label_dict]

    return sentences, sentences_span_label_dict


def process_document(document, logger_counter=None, max_length=1024, label_set=None):
    """
    Processing Document Entity-Relation-Event
    :param document:
    :param logger_counter ``Counter``
        Logger for Count Instances
    :param max_length ``int``
        Max Length
    :param label_set ``set``
        Valid Label Set
    :return:
        sentences ``List[List[Dict]]``
            Sentence is a list of token dicts
        clusters ``DefaultDict[int, List[Tuple[int, int]]]``
            1 -> [(span11_start, span11_end), (span12_start, span12_end), ...]
            2 -> [(span21_start, span22_end), (span22_start, span22_end), ...]
        sentences_span_label_dict ``List[Dict]``
            span_label_dict is a dict of token dicts
            (span_start, span_end) -> (event type label, event realis label)
    """
    token_to_index_map = dict()
    clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

    total_tokens = 0
    sentences = list()
    for sentence_index, sentence in enumerate(document['sentences']):

        for token_index, token in enumerate(sentence['tokens']):
            token_to_index_map[(sentence_index, token_index)] = total_tokens
            total_tokens += 1

            # Max Doc Length Skip
            if total_tokens >= max_length:
                break

        # Max Doc Length Skip
        if total_tokens > max_length:
            break

        sentences += [sentence['tokens']]

    # logger.info("event num: %s" % len(document['event']))
    span_label_dict = dict()
    for event_index, event in enumerate(document['event']):
        # logger.info("mention num: %s" % len(event['mentions']))
        for event_mention_index, event_mention in enumerate(event['mentions']):
            event_mention_label = "%s:%s" % (event_mention['type'], event_mention['subtype'])
            event_mention_realis = "%s" % event_mention.get('realis', 'NIL')

            if len(event_mention['nugget']['tokens']) == 0:
                if logger_counter is not None:
                    logger_counter.update(['mention escape: token not found'])
                continue

            if tuple(event_mention['nugget']['tokens'][0]) not in token_to_index_map:
                break
            if tuple(event_mention['nugget']['tokens'][-1]) not in token_to_index_map:
                break

            span_start = token_to_index_map[tuple(event_mention['nugget']['tokens'][0])]
            span_end = token_to_index_map[tuple(event_mention['nugget']['tokens'][-1])]
            span_key = (span_start, span_end)

            if label_set is not None and event_mention_label not in label_set:
                if logger_counter is not None:
                    logger_counter.update(['mention escape: label'])
                continue

            if span_key in span_label_dict:
                if logger_counter is not None:
                    logger_counter.update(['mention overlap'])
                continue

            span_label_dict[span_key] = (event_mention_label, event_mention_realis)

            if logger_counter is not None:
                logger_counter.update(['mention keep'])
            clusters[event_index].append((span_start, span_end))

    return sentences, clusters, span_label_dict


if __name__ == "__main__":
    pass
