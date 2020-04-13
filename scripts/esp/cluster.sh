#!/usr/bin/env bash

cd ~/novel-rl/

# GPU required algorithms
envlist=(
  BreakoutNoFrameskip-v4
  BeamRiderNoFrameskip-v4
  QbertNoFrameskip-v4
  SpaceInvadersNoFrameskip-v4
)

for envname in "${envlist[@]}"; do
  srun --gres=gpu:8 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
    --job-name=$algoname --mem=0 --exclusive \
    python scripts/esp/train_atari.py \
    --exp-name=esp_0413_atari \
    --env-name=$envname \
    --num-gpus=8 \
    --num-seeds=12 \
    2>&1 | tee esp_0413_atari_$envname.log &
done
