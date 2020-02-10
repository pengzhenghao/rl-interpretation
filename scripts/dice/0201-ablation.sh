#!/usr/bin/env bash

cd ~/novel-rl/

modename=ppo

srun --gres=gpu:0 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
  --job-name=$modename --mem=0 --exclusive \
  python scripts/0130-ppo-baseline-cpu.py \
  --exp-name=0130-ablation-$modename \
  --env-name=BipedalWalker-v2 \
  2>&1 | tee 0201-ablation-log/$modename.log &

modelist=(
constrain_novelty
use_bisector
two_side_clip_loss
normalize_advantage
delay_update
use_diversity_value_network
only_tnb
diversity_reward_type
)

for modename in "${modelist[@]}"; do
  srun --gres=gpu:0 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
    --job-name=$modename --mem=0 --exclusive \
    python scripts/0201-dece-bipedalwalker.py \
    --exp-name=0201-ablation-$modename \
    --mode=$modename \
    --num-gpus=0 \
    2>&1 | tee 0201-ablation-log/$modename.log &
done