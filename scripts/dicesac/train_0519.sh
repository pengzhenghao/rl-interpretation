cd ~/novel-rl

python toolbox/automation/launch.py \
  --exp-name dice_sac_0519 \
  --num-nodes 4 \
  --num-cpus 32 \
  --num-gpus 0 \
  --partition chpc \
  --command 'python scripts/dicesac/train.py --exp-name dice_sac_0519 --num-gpus 0 --num-seeds 2 ' \
