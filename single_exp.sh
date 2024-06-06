#!/bin/bash


sh scripts/exps/expand_diff.sh 5 0 0 &
sh scripts/exps/expand_diff.sh 5 1 1 &
sh scripts/exps/expand_diff.sh 5 2 2 &
sh scripts/exps/expand_diff.sh 5 3 3 &
wait

EXP=save/distdiff_batch_5x
sh scripts/exps/train_expand.sh caltech-101 resnet50 0.1 ${EXP} False 1
