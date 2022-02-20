# E3C

- An implementation for [End-to-End Neural Event Coreference Resolution](https://www.sciencedirect.com/science/article/pii/S0004370221001831)
- Please contact [Yaojie Lu](http://luyaojie.github.io) ([@luyaojie](mailto:yaojie2017@iscas.ac.cn)) for questions and suggestions.

## Quick links
* [Requirements](#Requirements)
* [Datasets](#Datasets)
* [Run Experiments](#Run-experiments)
* [Citation](#Citation)

## Requirements

General

- Python (verified on 3.7)
- CUDA (verified on 10.0)

Python Packages

- see requirements.txt

```bash
conda create -n event_coref python=3.7 -y
conda activate event_coref
pip install pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

Tools

- [CoreNLP](https://github.com/stanfordnlp/CoreNLP): for data pre-processing.
- [EvmEval](https://github.com/hunterhector/EvmEval): evaluation scorer.

``` bash
git clone https://github.com/hunterhector/EvmEval tools/EvmEval
```

Don't forget to star these repositories.

## Datasets

Datasets pre-processing details see [e3c_processing](https://drive.google.com/file/d/1CtoTL5CYQf-r-OP6aOfudtf69XKqYxXK/view?usp=sharing).

## Run Experiments

Parameters:
- `BERT_VERSION` pre-trained transformer local folder
- `SPAN_SIZE` max span size
- `-d` gpu device id
- `-c` config path
- `-m` trained model output path
- `-i` input data path
- `-k` run times
- `-o` extra parameter for config

KBP 2016 English
```bash
BERT_VERSION='/share/model/transformers/bert/uncased_L-12_H-768_A-12' SPAN_SIZE=1 \
  bash scripts/run_exp.bash \
  -d 0 \
  -c config/e3c_bert_base.jsonnet \
  -m model/e3c_bert_kbp2016_en \
  -i kbp_processing/data/data_split/jsonl_format/kbp2016/ \
  -k 3 \
  -o '{numpy_seed:42,pytorch_seed:42,random_seed:42}'
```

KBP 2017 English
```bash
BERT_VERSION='/share/model/transformers/bert/uncased_L-12_H-768_A-12' SPAN_SIZE=1 \
  bash scripts/run_exp.bash \
  -d 0 \
  -c config/e3c_bert_base.jsonnet \
  -m model/e3c_bert_kbp2017_en \
  -i kbp_processing/data/data_split/jsonl_format/kbp2017/ \
  -k 3 \
  -o '{numpy_seed:42,pytorch_seed:42,random_seed:42}'
```

KBP 2017 Chinese
```bash
BERT_VERSION='/share/model/transformers/bert/chinese_L-12_H-768_A-12' SPAN_SIZE=3 \
  bash scripts/run_exp.bash \
  -d 0 \
  -c config/e3c_bert_base.jsonnet \
  -m model/e3c_bert_kbp2017_zh \
  -i kbp_processing/data/data_split/jsonl_format/kbp2017_zh \
  -k 3 \
  -o '{numpy_seed:42,pytorch_seed:42,random_seed:42,model:{bce_loss_weight:10}}'
```

KBP 2017 Spanish
```bash
BERT_VERSION='/share/model/transformers/bert/beto_cased' SPAN_SIZE=1 \
  bash scripts/run_exp.bash \
  -d 0 \
  -c config/e3c_bert_base.jsonnet \
  -m model/e3c_bert_kbp2017_es \
  -i kbp_processing/data/data_split/jsonl_format/kbp2017_es \
  -k 3 \
  -o '{numpy_seed:42,pytorch_seed:42,random_seed:42}'
```

## Citation

If this repository helps you, please cite this paper:

Yaojie Lu, Hongyu Lin, Jialong Tang, Xianpei Han, Le Sun. End-to-End Neural Event Coreference Resolution. Artificial Intelligence, Volume 303, February 2022, 103632.

```
@article{LU:AIJ:2022:E3C,
  title = {End-to-end neural event coreference resolution},
  journal = {Artificial Intelligence},
  volume = {303},
  pages = {103632},
  year = {2022},
  issn = {0004-3702},
  doi = {https://doi.org/10.1016/j.artint.2021.103632},
  url = {https://www.sciencedirect.com/science/article/pii/S0004370221001831},
  author = {Yaojie Lu and Hongyu Lin and Jialong Tang and Xianpei Han and Le Sun},
  keywords = {Event coreference resolution, Event detection, End-to-end learning},
  abstract = {Conventional event coreference systems commonly use a pipeline architecture and rely heavily on handcrafted features, which often causes error propagation problems and leads to poor generalization ability. In this paper, we propose a neural network-based end-to-end event coreference architecture (E3C) that can jointly model event detection and event coreference resolution tasks and learn to extract features from raw text automatically. Furthermore, because event mentions are highly diversified and event coreference is intricately governed by long-distance and semantically-dependent decisions, a type-enhanced event coreference mechanism is further proposed in our E3C neural network. Experiments show that our method achieves a new state-of-the-art performance on both standard datasets.}
}
```
