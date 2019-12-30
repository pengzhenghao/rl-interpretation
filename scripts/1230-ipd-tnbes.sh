#!/usr/bin/env bash

ray start --head --redis-port 6789 --num-gpus 4

address="10.1.72.24:6789"

env_list=("Walker2d-v3" "Hopper-v3" "HalfCheetah-v3")

mkdir 1230-ipd-tnbes

for env in ${env_list[*]}; do
  expname="1230-ipd-tnbes-tnb-$env"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --address $address \
    --env-name $env \
    > 1230-ipd-tnbes-log/$expname.log 2>&1 &

  expname="1230-ipd-tnbes-ppo-$env"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --address $address \
    --env-name $env \
    --disable-tnb \
    > 1230-ipd-tnbes-log/$expname.log 2>&1 &
done