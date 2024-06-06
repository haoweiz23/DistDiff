#!/bin/bash
DATASET=$1
MODEL=$2
LR=$3
EXP=$4
Pretrained=$5
GPU=$6

for SEED in 1 2 3
do
    if [ "${Pretrained}" = "True" ]; then
      DIR=checkpoint/${DATASET}/${MODEL}_pretrained_${EXP}_lr${LR}/seed${SEED}
      if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
      else
          CUDA_VISIBLE_DEVICES=${GPU} python train_expanded_data_concat_original.py \
          -d ${DATASET} --checkpoint ${DIR} --data_dir data -a ${MODEL}  \
          --manualSeed ${SEED} --data_expanded_dir  data/${DATASET}_expansion/${EXP} \
          --pretrained --train-batch-size 64 --lr ${LR} --val-batch-size 64 --epochs 100
      fi

    else
      DIR=checkpoint/${DATASET}/${MODEL}_unpretrained_${EXP}_lr${LR}/seed${SEED}
      if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
      else
          CUDA_VISIBLE_DEVICES=${GPU} python train_expanded_data_concat_original.py \
          -d ${DATASET} --checkpoint ${DIR} --data_dir data -a ${MODEL}  \
          --manualSeed ${SEED} --data_expanded_dir  data/${DATASET}_expansion/${EXP} \
          --train-batch-size 64 --lr ${LR} --val-batch-size 64 --epochs 100
      fi
    fi


done