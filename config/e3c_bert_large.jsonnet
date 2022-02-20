// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “End-to-end Neural Coreference Resolution.” EMNLP (2017).
local data_path = std.extVar('DATA_FOLDER');
local bert_model = std.extVar('BERT_VERSION');
local max_span_width = std.parseInt(std.extVar('SPAN_SIZE'));
local bert_emb_size = 1024;
local span_emb_size = if max_span_width == 1 then bert_emb_size else (bert_emb_size * 3 + 20);
{
  dataset_reader: {
    type: 'event-coref-jsonl',
    token_indexers: {
      bert: {
        type: 'bert-pretrained',
        pretrained_model: bert_model,
        truncate_long_sequences: false,
      },
    },
    max_span_width: max_span_width,
    use_label_set: true,
  },
  train_data_path: data_path + '/train.jsonl',
  validation_data_path: data_path + '/valid.jsonl',
  model: {
    type: 'end-to-end-event-coreference',
    pretrain_ed: false,
    text_field_embedder: {
      allow_unmatched_keys: true,
      embedder_to_indexer_map: {
        bert: ['bert', 'bert-offsets'],
      },
      token_embedders: {
        bert: {
          type: 'bert-pretrained',
          pretrained_model: bert_model,
          top_layer_only: false,
          requires_grad: false,
        },
      },
    },
    mention_feedforward: {
      input_dim: span_emb_size,
      num_layers: 2,
      hidden_dims: 150,
      activations: 'relu',
      dropout: 0.2,
    },
    antecedent_feedforward: {
      input_dim: span_emb_size * 3 + 20,
      num_layers: 2,
      hidden_dims: 150,
      activations: 'relu',
      dropout: 0.2,
    },
    initializer: [
      ['.*linear_layers.*weight', { type: 'xavier_normal' }],
      ['.*scorer._module.weight', { type: 'xavier_normal' }],
      ['_distance_embedding.weight', { type: 'xavier_normal' }],
      ['_span_width_embedding.weight', { type: 'xavier_normal' }],
      ['_context_layer._module.weight_ih.*', { type: 'xavier_normal' }],
      ['_context_layer._module.weight_hh.*', { type: 'orthogonal' }],
    ],
    lexical_dropout: 0.5,
    feature_size: 20,
    max_span_width: max_span_width,
    spans_per_word: 0.1,
    max_antecedents: 50,
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['text', 'num_tokens']],
    padding_noise: 0.0,
    batch_size: 1,
  },
  trainer: {
    num_epochs: 150,
    grad_norm: 5.0,
    patience: 10,
    cuda_device: 0,
    validation_metric: '+a_f1',
    num_serialized_models_to_keep: 1,
    learning_rate_scheduler: {
      type: 'reduce_on_plateau',
      factor: 0.5,
      mode: 'max',
      patience: 5,
      verbose: true,
    },
    optimizer: {
      type: 'adamax',
    },
  },
}
