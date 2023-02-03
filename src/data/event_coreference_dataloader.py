#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019-09-11
import codecs
import collections
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from src.data.utils import kbp_label_set, canonicalize_clusters, process_document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("event-coref-jsonl")
class EventCoreferenceDatasetReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 max_span_width: int,
                 max_doc_length: int = 1024,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 use_label_set: bool = True) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._max_doc_length = max_doc_length
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_set = kbp_label_set if use_label_set else None
        logger.info(self._label_set)

    @overrides
    def _read(self, file_path: str):
        logger_counter = collections.Counter()
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with codecs.open(file_path, 'r', 'utf8') as fin:
            for doc_json in fin:
                document = json.loads(doc_json)
                sentences, clusters, span_label_dict = process_document(document,
                                                                        logger_counter,
                                                                        self._max_doc_length,
                                                                        label_set=self._label_set)
                canonical_clusters = canonicalize_clusters(clusters)
                yield self.text_to_instance(sentences, canonical_clusters, span_label_dict,
                                            doc_id=document.get('id', None))

        logger.info(logger_counter.__str__())

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[Dict]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
                         span_label_dict: Optional[Dict] = None,
                         doc_id: str = None,
                         ) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.
        span_label_dict: ``Optional[Dict[Tuple[int, int]: Tuple[str, str]]``, optional (default = None)
            A dict of label span in the document
            (start, end) -> (event label, event realis label)
        doc_id: str = None,

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        flattened_sentences = [self._normalize_word(token['word'])
                               for sentence in sentences
                               for token in sentence]
        token_offset = [(token['characterOffsetBegin'], token['characterOffsetEnd'])
                        for sentence in sentences
                        for token in sentence]

        if span_label_dict is not None:
            gold_label_spans = [(span[0], span[1], span_label_dict[span][0]) for span in span_label_dict]
        else:
            gold_label_spans = None

        metadata: Dict[str, Any] = {"original_text": flattened_sentences,
                                    "token_offset": token_offset,
                                    "span_label_dict": span_label_dict,
                                    "gold_label_spans": gold_label_spans
                                    }
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters
        if doc_id is not None:
            metadata["doc_id"] = doc_id

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_coref_labels: Optional[List[int]] = [] if gold_clusters is not None else None
        span_event_labels: Optional[List[int]] = [] if span_label_dict is not None else None
        span_realis_labels: Optional[List[int]] = [] if span_label_dict is not None else None

        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans([token['word'] for token in sentence],
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if span_coref_labels is not None:
                    if (start, end) in cluster_dict:
                        span_coref_labels.append(cluster_dict[(start, end)])
                    else:
                        span_coref_labels.append(-1)

                if span_label_dict is not None:
                    if (start, end) in span_label_dict:
                        event_label, event_realis = span_label_dict[(start, end)]
                        span_event_labels += [event_label]
                        span_realis_labels += [event_realis]
                    else:
                        span_event_labels += ['NIL']
                        span_realis_labels += ['NIL']

                spans.append(SpanField(start, end, text_field))

            sentence_offset += len(sentence)

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "spans": span_field,
                                    "metadata": metadata_field}

        if span_coref_labels is not None:
            fields["coref_labels"] = SequenceLabelField(span_coref_labels, span_field)
        if span_label_dict is not None:
            fields["event_type_labels"] = SequenceLabelField(span_event_labels, span_field, label_namespace="labels")
            fields["realis_labels"] = SequenceLabelField(span_realis_labels, span_field, label_namespace="tags")

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
