#!/bin/bash
DATASET=$1
MODEL=$2
LR=$3
Pretrained=$4
GPU=$5


for SEED in 1 2 3
do
    if [ "${Pretrained}" = "True" ]; then
        DIR=checkpoint/${DATASET}/${MODEL}_pretrained_lr${LR}/seed${SEED}
        if [ -d "$DIR" ]; then
          echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
          CUDA_VISIBLE_DEVICES=${GPU} python train.py -a ${MODEL} \
          -d ${DATASET} --checkpoint ${DIR} --data_dir data  \
          --manualSeed ${SEED} --pretrained \
          --train-batch-size 64 --lr ${LR} --val-batch-size 64 --epochs 100
        fi
    else
        DIR=checkpoint/${DATASET}/${MODEL}_unpretrained_lr${LR}/seed${SEED}
        if [ -d "$DIR" ]; then
          echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
          CUDA_VISIBLE_DEVICES=${GPU} python train.py -a ${MODEL} \
          -d ${DATASET} --checkpoint ${DIR} --data_dir data  \
          --manualSeed ${SEED} \
          --train-batch-size 64 --lr ${LR} --val-batch-size 64 --epochs 100
        fi
    fi
done
