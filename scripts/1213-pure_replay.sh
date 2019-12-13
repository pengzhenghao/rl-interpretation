# We have update our way of replay_values. In the new implementation,
# everything is replay, such as logp, prob, logit, instead of only replay the
# value.

nohup python toolbox/cooperative_exploration/train.py \
--exp-name 1213-pure_replay \
--mode two \
--test \
--num-seeds 1 \
> 1213-pure_replay.log 2>&1 &