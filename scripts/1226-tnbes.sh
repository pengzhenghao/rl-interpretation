#!/usr/bin/env bash

ray start --head --redis-port 6789 --num-gpus 4

address="10.1.72.24:6789"
th_list=(1.1 0.8)

for th in ${th_list[*]}; do
  expname="1226-tnbes-th$th-preoccupied"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --novelty-threshold $th \
    --use-preoccupied-agent \
    --address $address \
    --env-name Walker2d-v3 \
    > log/$expname.log 2>&1 &
done

for th in ${th_list[*]}; do
  expname="1226-tnbes-th$th-no-preoccupied"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --novelty-threshold $th \
    --address $address \
    --env-name Walker2d-v3 \
    > log/$expname.log 2>&1 &
done
