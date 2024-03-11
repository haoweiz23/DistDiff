#!/bin/bash

RHO=0.1
SCALE=50
STRENGTH=0.9        # 1.0 corresponds to full destruction of information in init image, default 0.9
EXPAND_NUM=5
INTERVAL=10000
DATASET=$1
START=$2
END=$3
CON=$4             # constraint_value
K=$5
GPU=$6
SPLIT=$7

DATA_SAVE_PATH=data/${DATASET}_expansion/distdiff_K${K}_CON_${CON}_start_${START}_end_${END}_batch_${EXPAND_NUM}x
CUDA_VISIBLE_DEVICES=${GPU} python3  data_expand.py \
        -a CLIP-VIT-B32   -d ${DATASET} \
        --data_dir data  \
        --data_save_dir ${DATA_SAVE_PATH} --opt_interval ${INTERVAL} --ckpt model/stable_diffusion_v1-4.ckpt  \
        --K ${K} --train-batch 1 --test-batch 1  --optimize_targets global_prototype local_prototype \
        --expanded_number_per_sample ${EXPAND_NUM} --expanded_batch_size 1 --start ${START} --end ${END} \
        --scale ${SCALE} --strength ${STRENGTH} --constraint_value ${CON} --rho ${RHO} --total_split 4 --split ${SPLIT}

