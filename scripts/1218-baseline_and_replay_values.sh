IP_ADDRESS=$1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python toolbox/cooperative_exploration/train.py \
--exp-name 1218-baseline_and_replay_values \
--mode baseline \
--num-gpus 4 --address $IP_ADDRESS \
> 1218-baseline_and_replay_values-baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 \
nohup python toolbox/cooperative_exploration/train.py \
--exp-name 1218-baseline_and_replay_values \
--mode replay_values \
--num-gpus 4 --address $IP_ADDRESS \
> 1218-baseline_and_replay_values-replay_values.log 2>&1 &