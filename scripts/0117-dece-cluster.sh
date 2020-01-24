#!/usr/bin/env bash

cd ~/novel-rl/

#envlist=(
#  'Walker2DBulletEnv-v0',
#  'HalfCheetahBulletEnv-v0',
#  'AntBulletEnv-v0',
#  'HopperBulletEnv-v0',
#)

envlist=(
  HalfCheetahBulletEnv-v0
  HopperBulletEnv-v0
)

for envname in "${envlist[@]}"; do
  srun --gres=gpu:K80:4 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
    --job-name=$envname --mem=0 --exclusive \
    python scripts/0117-dece-cluster.py \
    --exp-name=0117-dece-$envname \
    --env-name=$envname \
    --num-gpus=4 \
    2>&1 | tee log/$envname.log &
done
