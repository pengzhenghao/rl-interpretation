#!/usr/bin/env bash

address=$1
th_list=(0.4 0.5 0.6)

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
