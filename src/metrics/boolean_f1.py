#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019/10/12


from typing import Optional

import torch
from allennlp.training.metrics.metric import Metric
from overrides import overrides


@Metric.register("boolean_f1")
class BooleanF1(Metric):
    """
    只有 true 被计算
    Just checks batch-equality of two tensors and computes an accuracy metric based on that.
    That is, if your prediction has shape (batch_size, dim_1, ..., dim_n), this metric considers that
    as a set of `batch_size` predictions and checks that each is *entirely* correct across the remaining dims.
    This means the denominator in the accuracy computation is `batch_size`, with the caveat that predictions
    that are totally masked are ignored (in which case the denominator is the number of predictions that have
    at least one unmasked element).

    This is similar to :class:`CategoricalAccuracy`, if you've already done a ``.max()`` on your
    predictions.  If you have categorical output, though, you should typically just use
    :class:`CategoricalAccuracy`.  The reason you might want to use this instead is if you've done
    some kind of constrained inference and don't have a prediction tensor that matches the API of
    :class:`CategoricalAccuracy`, which assumes a final dimension of size ``num_classes``.
    """

    def __init__(self) -> None:
        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self._true_positive_sum = 0.
        # the total number of instances
        # Shape: (num_classes, )
        self._total_sum = 0.
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self._pred_sum = 0.
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self._true_sum = 0.

    def __call__(self,
                 predictions: torch.BoolTensor,
                 gold_labels: torch.BoolTensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ...).
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predictions``.
        mask: ``torch.Tensor``, optional (default = None).
            A tensor of the same shape as ``predictions``.
        """
        assert predictions.size() == gold_labels.size()
        if mask is not None:
            assert predictions.size() == mask.size()

        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)
        batch_size = predictions.size(0)

        if mask is not None:
            # We can multiply by the mask up front, because we're just checking equality below, and
            # this way everything that's masked will be equal.
            predictions = predictions * mask
            gold_labels = gold_labels * mask

            # We want to skip predictions that are completely masked;
            # so we'll keep predictions that aren't.
            keep = mask
        else:
            keep = torch.ones(predictions.size()).bool()

        predictions = predictions.view(batch_size, -1)
        gold_labels = gold_labels.view(batch_size, -1)

        # 只有 true 被计算
        self._true_positive_sum += torch.sum(predictions & gold_labels & keep).item()
        self._true_sum += torch.sum((predictions == gold_labels) & keep).item()
        self._pred_sum += torch.sum(predictions & keep).item()
        self._total_sum += torch.sum(gold_labels & keep).item()

    @staticmethod
    def safe_div(numerator, denominator):
        if denominator == 0.:
            return 0.
        else:
            return numerator / denominator

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        precision = self.safe_div(self._true_positive_sum, self._pred_sum)
        recall = self.safe_div(self._true_positive_sum, self._total_sum)
        f1 = self.safe_div(2 * precision * recall, precision + recall)
        metric = {'precision': precision,
                  'recall': recall,
                  'fscore': f1,
                  'tp': self._true_positive_sum,
                  'pred': self._pred_sum,
                  'gold': self._total_sum}
        if reset:
            self.reset()
        return metric

    @overrides
    def reset(self):
        self._true_positive_sum = 0.0
        self._true_sum = 0.0
        self._pred_sum = 0.0
        self._total_sum = 0.0


if __name__ == "__main__":
    pass
