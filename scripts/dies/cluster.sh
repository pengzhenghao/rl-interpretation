#!/usr/bin/env bash

cd ~/novel-rl/

# GPU required algorithms
algolist=(
  PPO
  DIES
)

for algoname in "${algolist[@]}"; do
  srun --gres=gpu:4 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
    --job-name=$algoname --mem=0 --exclusive \
    python scripts/dies/train_ppoes_dies.py \
    --exp-name=dies-0402-$algoname \
    --algo=$algoname \
    --num-gpus=4 \
    2>&1 | tee logs/dies/0402-$algoname.log &
done
