IP_ADDRESS=$1

nohup python toolbox/cooperative_exploration/train.py \
--exp-name 1218-baseline_and_replay_values \
--mode baseline \
--address $IP_ADDRESS \
> 1218-baseline_and_replay_values-baseline.log 2>&1 &

nohup python toolbox/cooperative_exploration/train.py \
--exp-name 1218-baseline_and_replay_values \
--mode replay_values \
--address $IP_ADDRESS \
> 1218-baseline_and_replay_values-replay_values.log 2>&1 &