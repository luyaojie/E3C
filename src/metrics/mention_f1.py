#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019/9/24
import logging
from typing import Any, Dict, List, Set, Tuple

import torch
from allennlp.training.metrics.metric import Metric
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Metric.register("mention_f1")
class MentionF1(Metric):
    def __init__(self) -> None:
        self._num_gold_mentions = 0
        self._num_predicted_mentions = 0
        self._num_recalled_mentions = 0

    @overrides
    def __call__(self,  # type: ignore
                 batched_top_spans: torch.Tensor,
                 batched_metadata: List[Dict[str, Any]]):
        for top_spans, metadata in zip(batched_top_spans.data.tolist(), batched_metadata):
            gold_mentions: Set[Tuple[int, int]] = {mention for cluster in metadata["clusters"]
                                                   for mention in cluster}
            predicted_spans: Set[Tuple[int, int]] = {(span[0], span[1]) for span in top_spans}
            self._num_gold_mentions += len(gold_mentions)
            self._num_recalled_mentions += len(gold_mentions & predicted_spans)
            self._num_predicted_mentions += len(predicted_spans)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict:
        if self._num_gold_mentions == 0:
            recall = 0.0
        else:
            recall = self._num_recalled_mentions / float(self._num_gold_mentions)

        if self._num_predicted_mentions == 0:
            precision = 0.0
        else:
            precision = self._num_recalled_mentions / float(self._num_predicted_mentions)

        if self._num_predicted_mentions != 0 and self._num_gold_mentions != 0 and self._num_recalled_mentions != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        if reset:
            self.reset()
        return {'precision': precision,
                'recall': recall,
                'f1-score': f1}

    @overrides
    def reset(self):
        self._num_predicted_mentions = 0
        self._num_gold_mentions = 0
        self._num_recalled_mentions = 0


@Metric.register("mention_type_f1")
class MentionTypeF1(MentionF1):
    def __init__(self, nil_index=0) -> None:
        super().__init__()
        self._nil_index = nil_index

    @overrides
    def __call__(self,  # type: ignore
                 event_type_predict_label: torch.Tensor,
                 event_type_labels: torch.Tensor,
                 span_mask: torch.Tensor):
        for predict_label, golden_label, mask in zip(event_type_predict_label, event_type_labels, span_mask):
            length = torch.sum(mask).long()
            predict_label = predict_label[:length]
            golden_label = golden_label[:length]

            self._num_gold_mentions += torch.sum(golden_label != self._nil_index).item()
            self._num_recalled_mentions += torch.sum(
                (predict_label == golden_label) & (golden_label != self._nil_index)).item()
            self._num_predicted_mentions += torch.sum(predict_label != self._nil_index).item()


@Metric.register("top_span_mention_type_f1")
class TopSpanMentionTypeF1(MentionF1):
    def __init__(self, nil_index=0) -> None:
        super().__init__()
        self._nil_index = nil_index

    @overrides
    def __call__(self,  # type: ignore
                 batch_pred_label_spans,
                 batch_gold_label_spans
                 ):
        for pred_label_spans, gold_label_spans in zip(batch_pred_label_spans, batch_gold_label_spans):
            predicted_spans = set(pred_label_spans)
            gold_mentions = set(gold_label_spans)
            self._num_gold_mentions += len(gold_mentions)
            self._num_recalled_mentions += len(gold_mentions & predicted_spans)
            self._num_predicted_mentions += len(predicted_spans)
