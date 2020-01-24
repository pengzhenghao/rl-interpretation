#!/bin/bash

#for envname in Walker2d-v3 Humanoid-v3 Hopper-v3 HalfCheetah-v3 Ant-v3

envname=Walker2d-v3

CUDA_VISIBLE_DEVICES=1,0 nohup python scripts/0123-dece-easy.py \
--env-name $envname --exp-name 0123-dece-easy-$envname \
> 0123-dece-easy-$envname.log 2>&1 &

envname=Humanoid-v3

CUDA_VISIBLE_DEVICES=1,2 nohup python scripts/0123-dece-easy.py \
--env-name $envname --exp-name 0123-dece-easy-$envname \
> 0123-dece-easy-$envname.log 2>&1 &

envname=HalfCheetah-v3

CUDA_VISIBLE_DEVICES=4,3 nohup python scripts/0123-dece-easy.py \
--env-name $envname --exp-name 0123-dece-easy-$envname \
> 0123-dece-easy-$envname.log 2>&1 &

envname=Hopper-v3

CUDA_VISIBLE_DEVICES=4,5 nohup python scripts/0123-dece-easy.py \
--env-name $envname --exp-name 0123-dece-easy-$envname \
> 0123-dece-easy-$envname.log 2>&1 &

envname=Ant-v3

CUDA_VISIBLE_DEVICES=7,6 nohup python scripts/0123-dece-easy.py \
--env-name $envname --exp-name 0123-dece-easy-$envname \
> 0123-dece-easy-$envname.log 2>&1 &

