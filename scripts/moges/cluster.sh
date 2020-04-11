#!/usr/bin/env bash

cd ~/novel-rl/

exp_name=0411_moges

srun \
  --gres=gpu:0 \
  -n1 \
  -p $MY_PARTITION_NAME \
  --kill-on-bad-exit=1 \
  --ntasks-per-node=1 \
  --job-name=$modename \
  --mem=0 \
  --exclusive \
  \
  \
  python toolbox/moges/train.py \
  --exp-name=$exp_name \
  --algo ES \
  --use-tanh \
  --num-gpus 0 \
  \
  \
  2>&1 | tee $exp_name.log &
