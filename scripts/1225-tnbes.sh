#!/usr/bin/env bash

ray start --head --redis-port 6789 --num-gpus 4

address="10.1.72.24:6789"
th_list=(0.5)

for th in ${th_list[*]}; do
  expname="1225-tnbes-th$th-preoccupied"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --novelty-threshold $th \
    --use-preoccupied-agent \
    --address $address \
    > log/$expname.log 2>&1 &
done

for th in ${th_list[*]}; do
  expname="1225-tnbes-th$th-no-preoccupied"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --novelty-threshold $th \
    --address $address \
    > log/$expname.log 2>&1 &
done
