#!/usr/bin/env bash

ray start --head --redis-port 6789 --num-gpus 4

address="10.1.72.24:6789"

env_list=("Walker2d-v3" "Hopper-v3" "HalfCheetah-v3")

mkdir 0102-ipd-tnbipdes-log

for env in ${env_list[*]}; do
  expname="0102-ipd-tnbipdes-tnb-$env"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb_es.py \
    --exp-name=$expname \
    --address $address \
    --env-name $env \
    > 0102-ipd-tnbipdes-log/$expname.log 2>&1 &

  expname="0102-ipd-tnbipdes-ipd-$env"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_ipd_es.py \
    --exp-name=$expname \
    --address $address \
    --env-name $env \
    > 0102-ipd-tnbipdes-log/$expname.log 2>&1 &
done