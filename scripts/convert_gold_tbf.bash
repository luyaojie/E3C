#!/usr/bin/env bash
# -*- coding:utf-8 -*-
# Created by Roger on 2019/9/25

KBP_DATA=~/Data/kbp_data
TBF_DATA=data/tbf_data

java -jar tools/EvmEval/bin/rich_ere_to_tbf_converter.jar \
  -a ${KBP_DATA}/LDC2017E55_TAC_KBP_2017_Eval_Core_Set_Rich_ERE_Annotation_with_Augmented_Event_Arguments/data/eng/nw/ere/ \
  --ae rich_ere.xml \
  -t ${KBP_DATA}/LDC2017E55_TAC_KBP_2017_Eval_Core_Set_Rich_ERE_Annotation_with_Augmented_Event_Arguments/data/eng/nw/source/ \
  --te xml \
  -o ${TBF_DATA}/LDC2017E55.nw.tbf

java -jar tools/EvmEval/bin/rich_ere_to_tbf_converter.jar \
  -a ${KBP_DATA}/LDC2017E55_TAC_KBP_2017_Eval_Core_Set_Rich_ERE_Annotation_with_Augmented_Event_Arguments/data/eng/df/ere/ \
  --ae rich_ere.xml \
  -t ${KBP_DATA}/LDC2017E55_TAC_KBP_2017_Eval_Core_Set_Rich_ERE_Annotation_with_Augmented_Event_Arguments/data/eng/df/source/ \
  --te xml \
  -o ${TBF_DATA}/LDC2017E55.df.tbf

cat ${TBF_DATA}/LDC2017E55.nw.tbf ${TBF_DATA}/LDC2017E55.df.tbf > ${TBF_DATA}/LDC2017E55.tbf
