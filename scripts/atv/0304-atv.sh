#!/usr/bin/env bash

cd ~/novel-rl/

# GPU required algorithms
algolist=(
  A2C
  A3C
  IMPALA
)
for algoname in "${algolist[@]}"; do
  srun --gres=gpu:4 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
    --job-name=$algoname --mem=0 --exclusive \
    python toolbox/atv/identical_parameters.py \
    --exp-name=0304-atv \
    --algo=$algoname \
    --num-gpus=4 \
    2>&1 | tee atv_logs/0304-$algoname.log &
done

# No GPU requirements algorithms
#algolist=(
#  ES
#  ARS
#)
#for algoname in "${algolist[@]}"; do
#  srun -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
#    --job-name=$algoname --mem=0 --exclusive \
#    python toolbox/atv/identical_parameters.py \
#    --exp-name=0304-atv \
#    --algo=$algoname \
#    --num-gpus=0 \
#    2>&1 | tee atv_logs/0304-$algoname.log &
#done
