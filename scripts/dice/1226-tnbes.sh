#!/usr/bin/env bash

ray start --head --redis-port 6789 --num-gpus 4

address="10.1.72.24:6789"

expname="1226-tnbes-preoccupied-ppo-walker"
echo "Start running: $expname"
nohup python toolbox/ipd/train_tnb.py \
  --exp-name=$expname \
  --use-preoccupied-agent \
  --address $address \
  --timesteps 2e6 \
  --env-name Walker2d-v3 \
  --disable-tnb \
  > log/$expname.log 2>&1 &

expname="1226-tnbes-no-preoccupied-ppo-walker"
echo "Start running: $expname"
nohup python toolbox/ipd/train_tnb.py \
  --exp-name=$expname \
  --address $address \
  --timesteps 5e6 \
  --max-not-improve-iterations 1 \
  --env-name Walker2d-v3 \
  --disable-tnb \
  > log/$expname.log 2>&1 &

expname="1226-tnbes-preoccupied-tnb-walker"
echo "Start running: $expname"
nohup python toolbox/ipd/train_tnb.py \
  --exp-name=$expname \
  --use-preoccupied-agent \
  --address $address \
  --timesteps 2e6 \
  --env-name Walker2d-v3 \
  > log/$expname.log 2>&1 &

expname="1226-tnbes-no-preoccupied-tnb-walker"
echo "Start running: $expname"
nohup python toolbox/ipd/train_tnb.py \
  --exp-name=$expname \
  --address $address \
  --timesteps 5e6 \
  --max-not-improve-iterations 1 \
  --env-name Walker2d-v3 \
  > log/$expname.log 2>&1 &