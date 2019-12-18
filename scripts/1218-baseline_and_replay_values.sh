CUDA_VISIBLE_DEVICES=1,2,3,4
nohup python toolbox/cooperative_exploration/train.py \
--exp-name 1218-baseline_and_replay_values \
--mode baseline \
> 1218-baseline_and_replay_values-baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7
nohup python toolbox/cooperative_exploration/train.py \
--exp-name 1218-baseline_and_replay_values \
--mode replay_values \
> 1218-baseline_and_replay_values-replay_values.log 2>&1 &