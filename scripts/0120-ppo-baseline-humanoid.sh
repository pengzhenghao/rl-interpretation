#!/usr/bin/env bash

cd ~/novel-rl/

#envlist=(
#  'Walker2DBulletEnv-v0',
#  'HalfCheetahBulletEnv-v0',
#  'AntBulletEnv-v0',
#  'HopperBulletEnv-v0',
#)

envlist=(
  HumanoidBulletEnv-v0
  HumanoidFlagrunBulletEnv-v0
)

for envname in "${envlist[@]}"; do
  srun --gres=gpu:4 -n1 --ntasks-per-node=1 \
    --job-name=$envname --mem=0 --exclusive \
    python scripts/0120-ppo-baseline-humanoid.py \
    --exp-name=0120-ppo-$envname \
    --env-name=$envname \
    --num-gpus=4 \
    2>&1 | tee log/0120-ppo-baseline-$envname.log &
done
