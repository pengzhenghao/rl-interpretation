#!/usr/bin/env bash

cd ~/novel-rl/

modename=ppo

srun --gres=gpu:0 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
  --job-name=$modename --mem=0 --exclusive \
  python scripts/0130-ppo-baseline-cpu.py \
  --exp-name=0204-ablation-$modename \
  --env-name=AntBulletEnv-v0 \
  2>&1 | tee 0204-ablation-log/$modename.log &

modelist=(
default
constrain_novelty
use_bisector
two_side_clip_loss
normalize_advantage
delay_update
use_diversity_value_network
only_tnb
)

for modename in "${modelist[@]}"; do
  srun --gres=gpu:4 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
    --job-name=$modename --mem=0 --exclusive \
    python scripts/0204-ablation.py \
    --exp-name=0204-ablation-$modename \
    --env-name=AntBulletEnv-v0 \
    --mode=$modename \
    2>&1 | tee 0204-ablation-log/$modename.log &
done