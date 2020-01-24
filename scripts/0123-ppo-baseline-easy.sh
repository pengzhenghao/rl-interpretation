#!/bin/bash

#for envname in Walker2d-v3 Humanoid-v3 Hopper-v3 HalfCheetah-v3 Ant-v3

envname=Walker2d-v3

CUDA_VISIBLE_DEVICES=1,0 nohup python scripts/0123-ppo-baseline-easy.py \
--env-name $envname --exp-name 0123-ppo-easy-$envname \
> 0123-ppo-easy-$envname.log 2>&1 &

envname=Humanoid-v3

CUDA_VISIBLE_DEVICES=1,0 nohup python scripts/0123-ppo-baseline-easy.py \
--env-name $envname --exp-name 0123-ppo-easy-$envname \
> 0123-ppo-easy-$envname.log 2>&1 &

envname=HalfCheetah-v3

CUDA_VISIBLE_DEVICES=0,1 nohup python scripts/0123-ppo-baseline-easy.py \
--env-name $envname --exp-name 0123-ppo-easy-$envname \
> 0123-ppo-easy-$envname.log 2>&1 &

envname=Hopper-v3

CUDA_VISIBLE_DEVICES=0,1 nohup python scripts/0123-ppo-baseline-easy.py \
--env-name $envname --exp-name 0123-ppo-easy-$envname \
> 0123-ppo-easy-$envname.log 2>&1 &

envname=Ant-v3

CUDA_VISIBLE_DEVICES=3,2 nohup python scripts/0123-ppo-baseline-easy.py \
--env-name $envname --exp-name 0123-ppo-easy-$envname \
> 0123-ppo-easy-$envname.log 2>&1 &

