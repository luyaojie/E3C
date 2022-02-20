#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019-09-10
# Mostly by AllenNLP

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Pruner
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import IntraSentenceAttentionEncoder
from allennlp.modules.similarity_functions import DotProductSimilarity
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average
from overrides import overrides
from torch.nn import BCEWithLogitsLoss

from src.metrics.event_coref_scores import EventCorefScores
from src.metrics.mention_f1 import TopSpanMentionTypeF1
from src.utils.cluster_decoding_utils import node_decode

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("end-to-end-event-coreference")
class End2EndEventCoreferenceResolver(Model):
    """
    This ``Model`` implements the coreference resolution model described "End-to-end Neural
    Coreference Resolution"
    <https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83>
    by Lee et al., 2017.
    The basic outline of this model is to get an embedded representation of each span in the
    document. These span representations are scored and used to prune away spans that are unlikely
    to occur in a coreference cluster. For the remaining spans, the model decides which antecedent
    span (if any) they are coreferent with. The resulting coreference links, after applying
    transitivity, imply a clustering of the spans in the document.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    antecedent_feedforward: ``FeedForward``
        This feedforward network is applied to pairs of span representation, along with any
        pairwise features, which is then scored by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    max_span_width: ``int``
        The maximum width of candidate spans.
    spans_per_word: float, required.
        A multiplier between zero and one which controls what percentage of candidate mention
        spans we retain with respect to the number of words in the document.
    max_antecedents: int, required.
        For each mention which survives the pruning stage, we consider this many antecedents.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 mention_feedforward: FeedForward,
                 antecedent_feedforward: FeedForward,
                 feature_size: int,
                 context_layer: Seq2SeqEncoder = None,
                 max_span_width: int = 1,
                 spans_per_word: float = 0.1,
                 max_antecedents: int = 50,
                 lexical_dropout: float = 0.2,
                 pretrain_ed: bool = False,
                 pretrain_coref: bool = False,
                 coref_loss_weight: float = 1.0,
                 bce_loss_weight: float = 1.0,
                 bce_pos_weight: float = None,
                 local_window_size: int = 10,
                 attention_type: str = 'dot',
                 decoding: str = 'type-guided',
                 type_threshold: float = -1.,
                 type_refine: bool = True,
                 type_match_in_eval: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(End2EndEventCoreferenceResolver, self).__init__(vocab, regularizer)
        logger.info(vocab)
        self._text_field_embedder = text_field_embedder
        self._context_layer = context_layer
        self._antecedent_feedforward = TimeDistributed(antecedent_feedforward)
        self._event_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1))
        )
        self._pretrain_ed = pretrain_ed
        self._pretrain_coref = pretrain_coref

        self._mention_pruner = Pruner(self._event_scorer)
        self._antecedent_scorer = TimeDistributed(torch.nn.Linear(antecedent_feedforward.get_output_dim(), 1))

        self._local_window_size = local_window_size
        self._attention_type = attention_type
        self._decoding = decoding
        self._type_threshold = type_threshold
        logger.info(vocab.get_token_from_index(0, "labels"))

        if context_layer is not None:
            endpoint_span_extractor_dim = context_layer.get_output_dim()
            attentive_span_extractor_dim = text_field_embedder.get_output_dim()
            self._endpoint_span_extractor = EndpointSpanExtractor(endpoint_span_extractor_dim,
                                                                  combination="x,y",
                                                                  num_width_embeddings=max_span_width,
                                                                  span_width_embedding_dim=feature_size)
            self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=attentive_span_extractor_dim)
            span_embedding_size = self._endpoint_span_extractor.get_output_dim() + self._attentive_span_extractor.get_output_dim()

            if self._local_window_size <= 0:
                self._attention_layer = None
            else:
                if self._attention_type == 'dot':
                    similarity_function = DotProductSimilarity(scale_output=True)
                    num_head = 1
                else:
                    raise NotImplementedError('Attention Type: %s' % self._attention_type)
                self._attention_layer = IntraSentenceAttentionEncoder(input_dim=attentive_span_extractor_dim,
                                                                      similarity_function=similarity_function,
                                                                      combination='2',
                                                                      num_attention_heads=num_head
                                                                      )
        else:
            attentive_span_extractor_dim = text_field_embedder.get_output_dim()

            if max_span_width > 1:
                endpoint_span_extractor_dim = text_field_embedder.get_output_dim()
                self._endpoint_span_extractor = EndpointSpanExtractor(endpoint_span_extractor_dim,
                                                                      combination="x,y",
                                                                      num_width_embeddings=max_span_width,
                                                                      span_width_embedding_dim=feature_size)
            else:
                self._endpoint_span_extractor = None

            self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=attentive_span_extractor_dim)

            if self._local_window_size <= 0:
                self._attention_layer = None
            else:
                if self._attention_type == 'dot':
                    similarity_function = DotProductSimilarity(scale_output=True)
                    num_head = 1
                else:
                    raise NotImplementedError('Attention Type: %s' % self._attention_type)
                self._attention_layer = IntraSentenceAttentionEncoder(input_dim=attentive_span_extractor_dim,
                                                                      similarity_function=similarity_function,
                                                                      combination='2',
                                                                      num_attention_heads=num_head
                                                                      )

            if self._endpoint_span_extractor is not None:
                span_embedding_size = self._attentive_span_extractor.get_output_dim() + self._endpoint_span_extractor.get_output_dim()
            else:
                span_embedding_size = self._attentive_span_extractor.get_output_dim()

        if type_refine:
            self._type_refine_gate = torch.nn.Sequential(
                TimeDistributed(torch.nn.Linear(span_embedding_size * 2, span_embedding_size)),
                torch.nn.Sigmoid()
            )
        else:
            self._type_refine_gate = None

        # NIL for Unified Event
        self._event_embedding = Embedding(num_embeddings=vocab.get_vocab_size('labels'),
                                          embedding_dim=span_embedding_size)
        self._event_embedding_map = torch.nn.Linear(self._event_embedding.get_output_dim() * 2,
                                                    self._event_embedding.get_output_dim())

        self._positive_label_size = vocab.get_vocab_size('labels') - 1

        # 10 possible distance buckets.
        self._num_distance_buckets = 10
        self._distance_embedding = Embedding(self._num_distance_buckets, feature_size)
        self._coref_loss_weight = coref_loss_weight
        self._bce_loss_weight = bce_loss_weight
        self._bce_pos_weight = bce_pos_weight

        self._max_span_width = max_span_width
        self._spans_per_word = spans_per_word
        self._max_antecedents = max_antecedents

        self._mention_f1_score = TopSpanMentionTypeF1()
        self._conll_coref_scores = EventCorefScores(mapping_type=type_match_in_eval)
        self._type_loss_metric = Average()
        self._realis_loss_metric = Average()
        self._coref_loss_metric = Average()
        self._coref_label_metric = Average()
        self._type_label_metric = Average()
        self._nil_label_metric = Average()

        if self._bce_pos_weight:
            self._bce_loss = BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(self._bce_pos_weight))
        else:
            self._bce_loss = BCEWithLogitsLoss(reduction='none')

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        initializer(self)

    def _get_event_embedding(self, span_mask):
        """
        :param span_mask:
            (batch, top_span_size, 1)
        :return:
            (batch, top_span_size, positive_label_size)
        """
        event_indices = util.get_range_vector(self._positive_label_size, device=util.get_device_of(span_mask)) + 1
        event_indices = torch.stack([torch.zeros_like(event_indices), event_indices]).transpose(0, 1)
        event_indices = event_indices.expand([event_indices.size(0), event_indices.size(1)])

        event_embeddings = self._event_embedding(event_indices)
        event_embeddings = event_embeddings.reshape(event_embeddings.size(0),
                                                    event_embeddings.size(1) * event_embeddings.size(2))

        event_embeddings = self._event_embedding_map.forward(event_embeddings)
        event_embeddings = event_embeddings.unsqueeze(0).expand(span_mask.size(0),
                                                                event_embeddings.size(0),
                                                                event_embeddings.size(1),
                                                                )
        return event_embeddings

    def _get_type_antecedent_labels(self, top_event_type_labels):
        """
        :param top_event_type_labels:
            (batch, top_span_size, 1)
        :return:
            (batch, top_span_size, positive_label_size)
        """
        event_indices = util.get_range_vector(self.vocab.get_vocab_size('labels'),
                                              device=util.get_device_of(top_event_type_labels))

        top_event_type_labels = top_event_type_labels.unsqueeze(-1).expand([top_event_type_labels.size(0),
                                                                            top_event_type_labels.size(1),
                                                                            event_indices.size(0)])

        type_antecedent_labels = (top_event_type_labels == event_indices).float()
        return type_antecedent_labels

    def _type_refine_embedding(self, top_embeddings, event_embeddings):
        # (batch, top_span_size, emb_size) bmm
        event_prob = torch.bmm(top_embeddings, torch.transpose(event_embeddings, 1, 2))
        shape = [event_prob.size(0), event_prob.size(1), 1]
        dummy_scores = event_prob.new_zeros(*shape)

        event_prob = torch.cat([dummy_scores, event_prob], -1)
        event_prob = torch.softmax(event_prob, -1)

        event_rep = torch.bmm(event_prob[:, :, 1:], event_embeddings) + event_prob[:, :, :1] * top_embeddings

        refine_gate = self._type_refine_gate(torch.cat([event_rep, top_embeddings], -1))

        top_embeddings = refine_gate * top_embeddings + (1 - refine_gate) * event_rep
        return top_embeddings

    def _local_attention(self, raw_contextualized_embeddings, text_mask):
        device = util.get_device_of(raw_contextualized_embeddings)
        if device < 0:
            device = 'cpu'
        attention_mask = torch.ones((text_mask.size(1), text_mask.size(1)), device=device)
        # attention_mask = attention_mask - torch.eye(text_mask.size(1),
        #                                             device=util.get_device_of(contextualized_embeddings))
        new_attention_mask = text_mask[:, :, None] * attention_mask
        new_attention_mask = torch.triu(torch.tril(new_attention_mask, self._local_window_size),
                                        -self._local_window_size)
        new_contextualized_embeddings = self._attention_layer(raw_contextualized_embeddings,
                                                              new_attention_mask)
        return new_contextualized_embeddings

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                spans: torch.IntTensor,
                coref_labels: torch.IntTensor = None,
                event_type_labels: torch.IntTensor = None,
                realis_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : ``Dict[str, torch.LongTensor]``, required.
            The output of a ``TextField`` representing the text of
            the document.
        spans : ``torch.IntTensor``, required.
            A tensor of shape (batch_size, num_spans, 2), representing the inclusive start and end
            indices of candidate spans for mentions. Comes from a ``ListField[SpanField]`` of
            indices into the text of the document.
        coref_labels : ``torch.IntTensor``, optional (default = None).
            A tensor of shape (batch_size, num_spans), representing the cluster ids
            of each span, or -1 for those which do not appear in any clusters.
        event_type_labels : ``torch.IntTensor``, optional (default = None).
            A tensor of shape (batch_size, num_spans), representing the event label of the specific span.
        realis_labels : ``torch.IntTensor``, optional (default = None).
            A tensor of shape (batch_size, num_spans), representing the realis label of the specific span.
        metadata : ``List[Dict[str, Any]]``, optional (default = None).
            A metadata dictionary for each instance in the batch. We use the "original_text" and "clusters" keys
            from this dictionary, which respectively have the original text and the annotated gold coreference
            clusters for that instance.

        Returns
        -------
        An output dictionary consisting of:
        top_spans : ``torch.IntTensor``
            A tensor of shape ``(batch_size, num_spans_to_keep, 2)`` representing
            the start and end word indices of the top spans that survived the pruning stage.
        antecedent_indices : ``torch.IntTensor``
            A tensor of shape ``(num_spans_to_keep, max_antecedents)`` representing for each top span
            the index (with respect to top_spans) of the possible antecedents the model considered.
        predicted_antecedents : ``torch.IntTensor``
            A tensor of shape ``(batch_size, num_spans_to_keep)`` representing, for each top span, the
            index (with respect to antecedent_indices) of the most likely antecedent. -1 means there
            was no predicted link.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        """
        # Shape: (batch_size, document_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))

        document_length = text_embeddings.size(1)
        num_spans = spans.size(1)

        # Shape: (batch_size, document_length)
        text_mask = util.get_text_field_mask(text).float()
        # Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).float()
        # SpanFields return -1 when they are used as padding. As we do
        # some comparisons based on span widths when we attend over the
        # span representations that we generate from these indices, we
        # need them to be <= 0. This is only relevant in edge cases where
        # the number of spans we consider after the pruning stage is >= the
        # total number of spans, because in this case, it is possible we might
        # consider a masked span.
        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        if self._context_layer:
            # Shape: (batch_size, document_length, encoding_dim)
            raw_contextualized_embeddings = self._context_layer(text_embeddings, text_mask)

            if self._attention_layer is not None:
                new_contextualized_embeddings = self._local_attention(
                    raw_contextualized_embeddings=raw_contextualized_embeddings,
                    text_mask=text_mask
                )
            else:
                new_contextualized_embeddings = raw_contextualized_embeddings

            # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
            endpoint_span_embeddings = self._endpoint_span_extractor(new_contextualized_embeddings, spans)
            # Shape: (batch_size, num_spans, embedding_size)
            attended_span_embeddings = self._attentive_span_extractor(text_embeddings, spans)

            # Shape: (batch_size, num_spans, embedding_size + 2 * encoding_dim + feature_size)
            # span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
            span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        else:
            raw_contextualized_embeddings = text_embeddings

            if self._attention_layer is not None:
                new_contextualized_embeddings = self._local_attention(
                    raw_contextualized_embeddings=raw_contextualized_embeddings,
                    text_mask=text_mask
                )
            else:
                new_contextualized_embeddings = raw_contextualized_embeddings

            span_embeddings_list = list()
            attended_span_embeddings = self._attentive_span_extractor(new_contextualized_embeddings, spans)
            span_embeddings_list += [attended_span_embeddings]
            if self._endpoint_span_extractor is not None:
                # Shape: (batch_size, num_spans, embedding_size)
                endpoint_span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)
                span_embeddings_list += [endpoint_span_embeddings]
            span_embeddings = torch.cat(span_embeddings_list, -1)

        # event_scores = self._event_classifier.forward(span_embeddings)
        # Shape: (batch_size, num_spans, num_event_realis_label)
        # Shape: (batch_size, num_spans, num_event_realis_label)
        # event_realis_scores = self._event_realis_classifier.forward(span_embeddings)

        # Prune based on mention scores.
        num_spans_to_keep_according_doc_len = int(math.floor(self._spans_per_word * document_length))

        (top_embeddings, top_mask, top_indices, top_scores) = self._mention_pruner(span_embeddings,
                                                                                   span_mask,
                                                                                   num_spans_to_keep_according_doc_len,
                                                                                   )

        event_embeddings = self._get_event_embedding(span_mask)
        top_mask = top_mask.unsqueeze(-1)
        # Shape: (batch_size * num_spans_to_keep)
        # torch.index_select only accepts 1D indices, but here
        # we need to select spans for each element in the batch.
        # This reformats the indices to take into account their
        # index into the batch. We precompute this here to make
        # the multiple calls to util.batched_index_select below more efficient.
        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_indices, num_spans)
        # Compute final predictions for which spans to consider as mentions.
        # Shape: (batch_size, num_spans_to_keep, 2)
        top_spans = util.batched_index_select(spans,
                                              top_indices,
                                              flat_top_span_indices)

        # Compute indices for antecedent spans to consider.
        max_antecedents = min(self._max_antecedents, num_spans_to_keep_according_doc_len)

        # top_span_embeddings = top_span_embeddings.detach()
        # top_span_mention_scores = top_span_mention_scores.detach()

        # Now that we have our variables in terms of num_spans_to_keep, we need to
        # compare span pairs to decide each span's antecedent. Each span can only
        # have prior spans as antecedents, and we only consider up to max_antecedents
        # prior spans. So the first thing we do is construct a matrix mapping a span's
        #  index to the indices of its allowed antecedents. Note that this is independent
        #  of the batch dimension - it's just a function of the span's position in
        # top_spans. The spans are in document order, so we can just use the relative
        # index of the spans to know which other spans are allowed antecedents.

        # Once we have this matrix, we reformat our variables again to get embeddings
        # for all valid antecedents for each span. This gives us variables with shapes
        #  like (batch_size, num_spans_to_keep, max_antecedents, embedding_size), which
        #  we can use to make coreference decisions between valid span pairs.

        # Shapes:
        # (num_spans_to_keep, max_antecedents),
        # (1, max_antecedents),
        # (1, num_spans_to_keep, max_antecedents)
        valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask = \
            _generate_valid_antecedents(num_spans_to_keep_according_doc_len,
                                        max_antecedents,
                                        util.get_device_of(text_mask))

        if self._type_refine_gate is not None:
            top_embeddings = self._type_refine_embedding(top_embeddings, event_embeddings)

        # Select tensors relating to the antecedent spans.
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        candidate_antecedent_embeddings = util.flattened_index_select(top_embeddings,
                                                                      valid_antecedent_indices)
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        candidate_antecedent_mention_scores = util.flattened_index_select(top_scores,
                                                                          valid_antecedent_indices).squeeze(-1)

        # Shape: (batch_size, num_spans_to_keep, event_type_size + max_antecedents, embedding_size)
        candidate_antecedent_embeddings = self._combine_event_embeddings_and_cluster_antecedent_embeddings(
            event_embeddings,
            candidate_antecedent_embeddings)

        # Compute antecedent scores.
        # Shape: (batch_size, num_spans_to_keep, event_type_size + max_antecedents, embedding_size)
        span_pair_embeddings = self._compute_span_pair_embeddings(top_embeddings,
                                                                  candidate_antecedent_embeddings,
                                                                  valid_antecedent_offsets)
        # (batch_size, event_type_size, 1)
        event_type_prior_scores = self._event_scorer(event_embeddings)
        # (batch_size, num_spans_to_keep, event_type_size)
        event_type_prior_scores = event_type_prior_scores.transpose(1, 2).expand(
            candidate_antecedent_mention_scores.size(0),
            candidate_antecedent_mention_scores.size(1),
            -1)

        # (batch_size, num_spans_to_keep, event_type_size + max_antecedents)
        candidate_antecedent_mention_scores = torch.cat([event_type_prior_scores,
                                                         candidate_antecedent_mention_scores],
                                                        -1)

        # Shape: (batch_size, num_spans_to_keep, 1 + event_type_size + max_antecedents)
        coreference_scores = self._compute_coreference_scores(span_pair_embeddings,
                                                              top_scores,
                                                              candidate_antecedent_mention_scores,
                                                              valid_antecedent_log_mask)

        # We now have, for each span which survived the pruning stage,
        # a predicted antecedent. This implies a clustering if we group
        # mentions which refer to each other in a chain.
        # Shape: (batch_size, num_spans_to_keep)
        _, predicted_antecedents = coreference_scores.max(2)
        # Subtract one here because index 0 is the "no antecedent" class,
        # so this makes the indices line up with actual spans if the prediction
        # is greater than -1.
        predicted_antecedents -= 1

        output_dict = {"top_spans": top_spans,
                       "antecedent_indices": valid_antecedent_indices,
                       "predicted_antecedents": predicted_antecedents,
                       "coreference_scores": coreference_scores,
                       }

        if coref_labels is not None and event_type_labels is not None:

            pruned_event_type_labels = torch.gather(event_type_labels, 1, top_indices)
            type_antecedent_labels = self._get_type_antecedent_labels(pruned_event_type_labels)

            # Find the gold labels for the spans which we kept.
            pruned_gold_labels = util.batched_index_select(coref_labels.unsqueeze(-1),
                                                           top_indices,
                                                           flat_top_span_indices)

            antecedent_labels = util.flattened_index_select(pruned_gold_labels,
                                                            valid_antecedent_indices).squeeze(-1)

            antecedent_labels += valid_antecedent_log_mask.long()

            # Compute labels.
            # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
            gold_antecedent_labels = self._compute_antecedent_gold_labels(pruned_gold_labels,
                                                                          type_antecedent_labels,
                                                                          antecedent_labels)

            bce_loss = self._bce_loss.forward(self._event_scorer.forward(span_embeddings).squeeze(-1),
                                              (event_type_labels > 0).float()) * span_mask
            bce_loss = bce_loss.sum() * self._bce_loss_weight

            # Now, compute the loss using the negative marginal log-likelihood.
            # This is equal to the log of the sum of the probabilities of all antecedent predictions
            # that would be consistent with the data, in the sense that we are minimising, for a
            # given span, the negative marginal log likelihood of all antecedents which are in the
            # same gold cluster as the span we are currently considering. Each span i predicts a
            # single antecedent j, but there might be several prior mentions k in the same
            # coreference cluster that would be valid antecedents. Our loss is the sum of the
            # probability assigned to all valid antecedents. This is a valid objective for
            # clustering as we don't mind which antecedent is predicted, so long as they are in
            #  the same coreference cluster.

            if self._pretrain_ed:
                # All antecedent mask is 0
                top_mask = top_mask.expand_as(coreference_scores).clone()
                top_mask[:, :, self._positive_label_size + 2:] = 0

            coreference_log_probs = util.masked_log_softmax(coreference_scores, top_mask)
            correct_antecedent_log_probs = coreference_log_probs + gold_antecedent_labels.log()
            negative_marginal_log_likelihood = -util.logsumexp(correct_antecedent_log_probs).sum()
            coref_loss = negative_marginal_log_likelihood * self._coref_loss_weight

            output_dict["loss"] = coref_loss + bce_loss

            decoded_result = self.decode(output_dict)

            pred_label_spans_list = decoded_result['pred_label_spans']
            gold_label_spans_list = [m['gold_label_spans'] for m in metadata]

            self._mention_f1_score(pred_label_spans_list,
                                   gold_label_spans_list,
                                   )
            self._conll_coref_scores(decoded_result['clusters'],
                                     metadata,
                                     pred_label_spans_list,
                                     gold_label_spans_list)

            self._type_loss_metric(bce_loss.item())
            self._coref_loss_metric(negative_marginal_log_likelihood.item())
        else:
            self._coref_loss_metric(0.)

        if metadata is not None:
            output_dict["document"] = [x["original_text"] for x in metadata]
            output_dict["offset"] = [x["token_offset"] for x in metadata]
            output_dict['doc_id'] = [x.get("doc_id", None) for x in metadata]

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.

        Returns
        -------
        The same output dictionary, but with an additional ``clusters`` key:

        clusters : ``List[List[List[Tuple[int, int]]]]``
            A nested list, representing, for each instance in the batch, the list of clusters,
            which are in turn comprised of a list of (start, end) inclusive spans into the
            original document.
        """
        return node_decode(output_dict,
                           self.vocab, decoding_algorithm=self._decoding,
                           positive_label_size=self._positive_label_size,
                           type_threshold=self._type_threshold)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        mention_result = self._mention_f1_score.get_metric(reset)
        coref_precision, coref_recall, coref_f1 = self._conll_coref_scores.get_metric(reset)

        return {"c_p": coref_precision,
                "c_r": coref_recall,
                "c_f1": coref_f1,
                "m_p": mention_result['precision'],
                "m_r": mention_result['recall'],
                "m_f1": mention_result['f1-score'],
                "nil": self._nil_label_metric.get_metric(reset),
                "type": self._type_label_metric.get_metric(reset),
                "coref": self._coref_label_metric.get_metric(reset),
                "t_l": self._type_loss_metric.get_metric(reset),
                "c_l": self._coref_loss_metric.get_metric(reset),
                "a_f1": (mention_result['f1-score'] + coref_f1) / 2.}

    @staticmethod
    def _combine_event_embeddings_and_cluster_antecedent_embeddings(event_embeddings: torch.FloatTensor,
                                                                    antecedent_embeddings: torch.FloatTensor):
        """
        event_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of the event types. Has shape
            (batch_size, event_type_size, embedding_size).
        antecedent_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size).
        return:
            (batch_size, num_spans_to_keep, max_antecedents + event_type_size, embedding_size)
        """
        event_embeddings = event_embeddings.unsqueeze(1).expand((antecedent_embeddings.size(0),
                                                                 antecedent_embeddings.size(1),
                                                                 event_embeddings.size(1),
                                                                 antecedent_embeddings.size(3),))
        return torch.cat([event_embeddings, antecedent_embeddings], 2)

    def _compute_span_pair_embeddings(self,
                                      top_span_embeddings: torch.FloatTensor,
                                      antecedent_embeddings: torch.FloatTensor,
                                      antecedent_offsets: torch.FloatTensor):
        """
        Computes an embedding representation of pairs of spans for the pairwise scoring function
        to consider. This includes both the original span representations, the element-wise
        similarity of the span representations, and an embedding representation of the distance
        between the two spans.

        Parameters
        ---------- shape
            (batch_size, event_type_size, embedding_size).
        top_span_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the top spans. Has shape
            (batch_size, num_spans_to_keep, embedding_size).
        antecedent_embeddings : ``torch.FloatTensor``, required.
            Embedding representations of the antecedent spans we are considering
            for each top span. Has shape
            (batch_size, num_spans_to_keep, event_type_size + max_antecedents, embedding_size).
        antecedent_offsets : ``torch.IntTensor``, required.
            The offsets between each top span and its antecedent spans in terms
            of spans we are considering. Has shape (1, max_antecedents).

        Returns
        -------
        span_pair_embeddings : ``torch.FloatTensor``
            Embedding representation of the pair of spans to consider. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents, embedding_size)
        target_embeddings = top_span_embeddings.unsqueeze(2).expand_as(antecedent_embeddings)

        # Shape: (1, max_antecedents)
        bucket_values = util.bucket_values(antecedent_offsets, num_total_buckets=self._num_distance_buckets)
        # (1, event_type)
        label_bucket_values = bucket_values.new_zeros((1, self._positive_label_size))
        # Shape: (1, max_antecedents + event_type_size, embedding_size)
        antecedent_distance_embeddings = self._distance_embedding(
            torch.cat([bucket_values, label_bucket_values], 1)
        )

        # Shape: (1, 1, max_antecedents + event_type_size, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.unsqueeze(0)

        expanded_distance_embeddings_shape = (antecedent_embeddings.size(0),
                                              antecedent_embeddings.size(1),
                                              antecedent_embeddings.size(2),
                                              antecedent_distance_embeddings.size(-1))
        # Shape: (batch_size, num_spans_to_keep, max_antecedents + event_type_size, embedding_size)
        antecedent_distance_embeddings = antecedent_distance_embeddings.expand(*expanded_distance_embeddings_shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + event_type_size, embedding_size)
        span_pair_embeddings = torch.cat([target_embeddings,
                                          antecedent_embeddings,
                                          antecedent_embeddings * target_embeddings,
                                          antecedent_distance_embeddings], -1)
        return span_pair_embeddings

    def _compute_antecedent_gold_labels(self,
                                        top_span_labels: torch.IntTensor,
                                        type_antecedent_labels: torch.IntTensor,
                                        antecedent_labels: torch.IntTensor):
        """
        Generates a binary indicator for every pair of spans. This label is one if and
        only if the pair of spans belong to the same cluster. The labels are augmented
        with a dummy antecedent at the zeroth position, which represents the prediction
        that a span does not have any antecedent.

        Parameters
        ----------
        top_span_labels : ``torch.IntTensor``, required.
            The cluster id label for every span. The id is arbitrary,
            as we just care about the clustering. Has shape (batch_size, num_spans_to_keep).
        antecedent_labels : ``torch.IntTensor``, required.
            The cluster id label for every antecedent span. The id is arbitrary,
            as we just care about the clustering. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).

        Returns
        -------
        pairwise_labels_with_dummy_label : ``torch.FloatTensor``
            A binary tensor representing whether a given pair of spans belong to
            the same cluster in the gold clustering.
            Has shape (batch_size, num_spans_to_keep, max_antecedents + 1).

        """
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        # print(top_span_labels)
        # print(antecedent_labels)

        target_labels = top_span_labels.expand_as(antecedent_labels)
        same_cluster_indicator = (target_labels == antecedent_labels).float()
        non_dummy_indicator = (target_labels >= 0).float()
        pairwise_labels = same_cluster_indicator * non_dummy_indicator

        if self._pretrain_ed:
            pairwise_labels = pairwise_labels * 0
        else:
            # for pairwise_labels without type_antecedent_labels
            pairwise_labels_indicator = (pairwise_labels.sum(-1, keepdim=True) > 0).float()
            type_antecedent_labels = type_antecedent_labels * (1 - pairwise_labels_indicator)

        self._coref_label_metric(torch.sum(pairwise_labels).item())
        self._nil_label_metric(torch.sum(type_antecedent_labels[:, :, 0]).item())
        self._type_label_metric(torch.sum(type_antecedent_labels[:, :, 1: self._positive_label_size + 1]).item())

        # print(pairwise_labels)
        #
        # # Shape: (batch_size, num_spans_to_keep, 1)
        # dummy_labels = (1 - pairwise_labels).prod(-1, keepdim=True)

        # Shape: (batch_size, num_spans_to_keep, event_type_size + max_antecedents + 1)
        pairwise_labels_with_dummy_label = torch.cat([type_antecedent_labels, pairwise_labels], -1)
        return pairwise_labels_with_dummy_label

    def _compute_coreference_scores(self,
                                    pairwise_embeddings: torch.FloatTensor,
                                    top_span_mention_scores: torch.FloatTensor,
                                    antecedent_mention_scores: torch.FloatTensor,
                                    antecedent_log_mask: torch.FloatTensor) -> torch.FloatTensor:
        """
        Computes scores for every pair of spans. Additionally, a dummy label is included,
        representing the decision that the span is not coreferent with anything. For the dummy
        label, the score is always zero. For the true antecedent spans, the score consists of
        the pairwise antecedent score and the unary mention scores for the span and its
        antecedent. The factoring allows the model to blame many of the absent links on bad
        spans, enabling the pruning strategy used in the forward pass.

        Parameters
        ----------
        pairwise_embeddings: ``torch.FloatTensor``, required.
            Embedding representations of pairs of spans. Has shape
            (batch_size, num_spans_to_keep, max_antecedents, encoding_dim)
        top_span_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every span. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_mention_scores: ``torch.FloatTensor``, required.
            Mention scores for every antecedent. Has shape
            (batch_size, num_spans_to_keep, max_antecedents).
        antecedent_log_mask: ``torch.FloatTensor``, required.
            The log of the mask for valid antecedents.

        Returns
        -------
        coreference_scores: ``torch.FloatTensor``
            A tensor of shape (batch_size, num_spans_to_keep, max_antecedents + 1),
            representing the unormalised score for each (span, antecedent) pair
            we considered.

        """
        antecedent_log_mask = torch.cat([antecedent_log_mask.new_zeros((antecedent_log_mask.size(0),
                                                                        antecedent_log_mask.size(1),
                                                                        self._positive_label_size)),
                                         antecedent_log_mask],
                                        -1)
        # Shape: (batch_size, num_spans_to_keep, max_antecedents)
        antecedent_scores = self._antecedent_scorer(
            self._antecedent_feedforward(pairwise_embeddings)).squeeze(-1)
        antecedent_scores += top_span_mention_scores + antecedent_mention_scores
        antecedent_scores += antecedent_log_mask

        # Shape: (batch_size, num_spans_to_keep, 1)
        shape = [antecedent_scores.size(0), antecedent_scores.size(1), 1]
        dummy_scores = antecedent_scores.new_zeros(*shape)

        # Shape: (batch_size, num_spans_to_keep, max_antecedents + 1)
        coreference_scores = torch.cat([dummy_scores, antecedent_scores], -1)
        return coreference_scores


def _generate_valid_antecedents(num_spans_to_keep: int,
                                max_antecedents: int,
                                device: int) -> Tuple[torch.IntTensor,
                                                      torch.IntTensor,
                                                      torch.FloatTensor]:
    """
    This method generates possible antecedents per span which survived the pruning
    stage. This procedure is `generic across the batch`. The reason this is the case is
    that each span in a batch can be coreferent with any previous span, but here we
    are computing the possible `indices` of these spans. So, regardless of the batch,
    the 1st span _cannot_ have any antecedents, because there are none to select from.
    Similarly, each element can only predict previous spans, so this returns a matrix
    of shape (num_spans_to_keep, max_antecedents), where the (i,j)-th index is equal to
    (i - 1) - j if j <= i, or zero otherwise.

    Parameters
    ----------
    num_spans_to_keep : ``int``, required.
        The number of spans that were kept while pruning.
    max_antecedents : ``int``, required.
        The maximum number of antecedent spans to consider for every span.
    device: ``int``, required.
        The CUDA device to use.

    Returns
    -------
    valid_antecedent_indices : ``torch.IntTensor``
        The indices of every antecedent to consider with respect to the top k spans.
        Has shape ``(num_spans_to_keep, max_antecedents)``.
    valid_antecedent_offsets : ``torch.IntTensor``
        The distance between the span and each of its antecedents in terms of the number
        of considered spans (i.e not the word distance between the spans).
        Has shape ``(1, max_antecedents)``.
    valid_antecedent_log_mask : ``torch.FloatTensor``
        The logged mask representing whether each antecedent span is valid. Required since
        different spans have different numbers of valid antecedents. For example, the first
        span in the document should have no valid antecedents.
        Has shape ``(1, num_spans_to_keep, max_antecedents)``.
    """
    # Shape: (num_spans_to_keep, 1)
    target_indices = util.get_range_vector(num_spans_to_keep, device).unsqueeze(1)

    # Shape: (1, max_antecedents)
    valid_antecedent_offsets = (util.get_range_vector(max_antecedents, device) + 1).unsqueeze(0)

    # This is a broadcasted subtraction.
    # Shape: (num_spans_to_keep, max_antecedents)
    raw_antecedent_indices = target_indices - valid_antecedent_offsets

    # In our matrix of indices, the upper triangular part will be negative
    # because the offsets will be > the target indices. We want to mask these,
    # because these are exactly the indices which we don't want to predict, per span.
    # We're generating a logspace mask here because we will eventually create a
    # distribution over these indices, so we need the 0 elements of the mask to be -inf
    # in order to not mess up the normalisation of the distribution.
    # Shape: (1, num_spans_to_keep, max_antecedents)
    valid_antecedent_log_mask = (raw_antecedent_indices >= 0).float().unsqueeze(0).log()

    # Shape: (num_spans_to_keep, max_antecedents)
    valid_antecedent_indices = F.relu(raw_antecedent_indices.float()).long()
    return valid_antecedent_indices, valid_antecedent_offsets, valid_antecedent_log_mask
