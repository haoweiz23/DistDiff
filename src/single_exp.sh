#!/bin/bash

for DATASET in caltech-101
do
  START=32
  END=30
  sh scripts/exps/expand_diff.sh ${DATASET} ${START} ${END} 0.8 10000 3 1 0 &
  sh scripts/exps/expand_diff.sh ${DATASET} ${START} ${END} 0.8 10000 3 2 1 &
  sh scripts/exps/expand_diff.sh ${DATASET} ${START} ${END} 0.8 10000 3 3 2 &
  sh scripts/exps/expand_diff.sh ${DATASET} ${START} ${END} 0.8 10000 3 4 3 &
  wait

  EXP=distdiff_K3_CON_0.8_start_${START}_end_${END}_interval_10000_batch_5x
  sh scripts/exps/train_expand.sh ${DATASET} resnet50 0.1 ${EXP} False 1 &
done
