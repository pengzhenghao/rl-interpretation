#!/usr/bin/env bash

cd ~/novel-rl/

exp_name=0411_moges

srun \
  --gres=gpu:0 \
  -n1 \
  --kill-on-bad-exit=1 \
  --ntasks-per-node=1 \
  --job-name=$modename \
  --mem=0 \
  --exclusive \
  \
  \
  python scripts/0130-ppo-baseline-cpu.py \
  --exp-name=$exp_name \
  --num-gpus 0 \
  \
  \
  2>&1 | tee $exp_name.log &
