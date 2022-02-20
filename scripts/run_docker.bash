#!/usr/bin/env bash
# -*- coding:utf-8 -*- 

docker run -t -i \
  --name event_coref_exp_20200408 \
  --gpus all \
  -v /share:/share \
  -v /home/yaojie/work/event_coref_exp:/event_coref_exp \
  allennlp_transformers:latest \
  /bin/bash
