#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Created by Roger on 2019-08-14

EXP_ID=$(date +%F-%H-%M-$RANDOM)
DEVICE=0
DATA_FOLDER=data/kbp_zh
CONFIG_PATH=config/e3c/e3c_bert_base.jsonnet
EXP_PREFIX=model/e3c_bert_base
OVERRIDES='{}'
RUN_TIME=1

while getopts "d:m:c:o:k:i:" arg; do #选项后面的冒号表示该选项需要参数
  case $arg in
  d)
    DEVICE=$OPTARG
    ;;
  c)
    CONFIG_PATH=$OPTARG
    ;;
  m)
    EXP_PREFIX=$OPTARG
    ;;
  o)
    OVERRIDES=$OPTARG
    ;;
  k)
    RUN_TIME=$OPTARG
    ;;
  i)
    DATA_FOLDER=$OPTARG
    ;;
  ?) #当有不认识的选项的时候arg为?
    echo "unkonw argument"
    exit 1
    ;;
  esac
done

EXP_MODEL_NAME=${EXP_PREFIX}_${EXP_ID}

export DATA_FOLDER=${DATA_FOLDER}

echo "Save to ${EXP_MODEL_NAME}"

for run_time in $(seq 1 ${RUN_TIME}); do

  CUDA_VISIBLE_DEVICES=${DEVICE} allennlp train \
    -s ${EXP_MODEL_NAME}_run${run_time} \
    --include-package src \
    --file-friendly-logging \
    --overrides ${OVERRIDES} \
    ${CONFIG_PATH}

  for data in valid test; do
    bash scripts/run_eval.bash -m ${EXP_MODEL_NAME}_run${run_time} -d ${DEVICE} -i ${DATA_FOLDER}/${data}.jsonl -t ${DATA_FOLDER}/${data}.tbf -n ${data}
  done

done

echo "Save to " ${EXP_MODEL_NAME}
