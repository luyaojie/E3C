#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Created by Roger on 2019-09-25

MODEL_PATH=""
DATA_PATH="data/split_yin_jou_huang_naacl2019_50dev/test.jsonl"
DEVICE=0
TBF_PATH="data/tbf_data/LDC2017E55.tbf"
WAIT=0
PREDICTOR="event-coref"
PREFIX="event-coref"
WEIGHT_PATH="best.th"

while getopts "d:m:i:t:p:w:n:" arg; do #选项后面的冒号表示该选项需要参数
  case $arg in
  d)
    DEVICE=$OPTARG
    ;;
  m)
    MODEL_PATH=$OPTARG
    ;;
  i)
    DATA_PATH=$OPTARG
    ;;
  t)
    TBF_PATH=$OPTARG
    ;;
  p)
    PREDICTOR=$OPTARG
    ;;
  n)
    PREFIX=$OPTARG
    ;;
  w)
    WEIGHT_PATH=$OPTARG
    ;;
  ?) #当有不认识的选项的时候arg为?
    echo "unkonw argument"
    exit 1
    ;;
  esac
done

if [ -z $MODEL_PATH ]; then
  echo "Model Path -m needed!" && exit 1
fi
if [ -z $DATA_PATH ]; then
  echo "Data Path -i needed!" && exit 1
fi

allennlp predict \
  ${MODEL_PATH} \
  ${DATA_PATH} \
  --output-file ${MODEL_PATH}/${PREFIX}-${PREDICTOR}-predict.jsonl \
  --cuda-device ${DEVICE} \
  --weights-file ${MODEL_PATH}/${WEIGHT_PATH} \
  --include-package src --silent --predictor ${PREDICTOR}

python3 ./scripts/convert_predict_to_tbf.py \
  ${MODEL_PATH}/${PREFIX}-${PREDICTOR}-predict.jsonl \
  ${MODEL_PATH}/${PREFIX}-predict.tbf

if [ ${WAIT} == "1" ]; then
  read -p "WAIT" choice
fi

python2 tools/EvmEval/scorer_v1.8.py \
  -g ${TBF_PATH} \
  -s ${MODEL_PATH}/${PREFIX}-predict.tbf \
  -c ${MODEL_PATH}/conll_coref \
  -m 0 >${MODEL_PATH}/${PREFIX}_notyped-eval.result

python2 tools/EvmEval/scorer_v1.8.py \
  -g ${TBF_PATH} \
  -s ${MODEL_PATH}/${PREFIX}-predict.tbf \
  -c ${MODEL_PATH}/conll_coref \
  -m 1 >${MODEL_PATH}/${PREFIX}_typed-eval.result

tail -n 20 ${MODEL_PATH}/${PREFIX}_*-eval.result

echo 'Match Type: ' ${MATCH_TYPE}
echo 'Model Path: ' ${MODEL_PATH}
echo 'Data  Path: ' ${DATA_PATH}
echo 'TBF   Path: ' ${TBF_PATH}
echo 'Predictor : ' ${PREDICTOR}
echo 'WAIT: ' ${WAIT}
