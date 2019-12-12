# We have update our way of replay_values. In the new implementation,
# everything is replay, such as logp, prob, logit, instead of only replay the
# value.

expname="1213-pure_replay"

srun --gres=gpu:8 -n1 --kill-on-bad-exit=1 --ntasks-per-node=1 \
  --job-name=$expname --mem=300G \
  python toolbox/cooperative_exploration/train.py \
  --exp-name $expname \
  --mode three_baselines \
  2>&1 | tee $expname.log &
