#!/usr/bin/env bash

ray start --head --redis-port 6789 --num-gpus 4

address="10.1.72.24:6789"

th_list=(0.4 0.8)

mkdir 1127-tnbes-log

for th in ${th_list[*]}; do
  expname="1227-tnbes-preoccupied-tnb-plus-walker-th$th"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --use-preoccupied-agent \
    --address $address \
    --timesteps 2e6 \
    --env-name Walker2d-v3 \
    --novelty-threshold $th \
    --use-tnb-plus \
    > 1227-tnbes-log/$expname.log 2>&1 &

  expname="1227-tnbes-no-preoccupied-tnb-plus-walker-th$th"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --address $address \
    --timesteps 5e6 \
    --max-not-improve-iterations 1 \
    --env-name Walker2d-v3 \
    --novelty-threshold $th \
    --use-tnb-plus \
    > 1227-tnbes-log/$expname.log 2>&1 &

  expname="1227-tnbes-preoccupied-tnb-walker-th$th"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --use-preoccupied-agent \
    --address $address \
    --timesteps 2e6 \
    --env-name Walker2d-v3 \
    --novelty-threshold $th \
    > 1227-tnbes-log/$expname.log 2>&1 &

  expname="1227-tnbes-no-preoccupied-tnb-walker-th$th"
  echo "Start running: $expname"
  nohup python toolbox/ipd/train_tnb.py \
    --exp-name=$expname \
    --address $address \
    --timesteps 5e6 \
    --max-not-improve-iterations 1 \
    --env-name Walker2d-v3 \
    --novelty-threshold $th \
    > 1227-tnbes-log/$expname.log 2>&1 &
done