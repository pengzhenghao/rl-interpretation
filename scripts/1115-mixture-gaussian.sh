# Please call this script at novel-rl directory.
ulimit -u 65535
ulimit -n 65535

exp_name="1115-mixture_gaussian"

env_name="BipedalWalker-v2"
CUDA_VISIBLE_DEVICES=1 nohup python \
toolbox/action_distribution/train_mixture_gaussian.py \
--exp-name ${exp_name}_${env_name} --num-gpus 1 --env ${env_name} \
--num-seeds 1 > ${exp_name}_${env_name}.log 2>&1 &

env_name="Walker2d-v3"
CUDA_VISIBLE_DEVICES=2 nohup python \
toolbox/action_distribution/train_mixture_gaussian.py \
--exp-name ${exp_name}_${env_name} --num-gpus 1 --env ${env_name} \
--num-seeds 1 > ${exp_name}_${env_name}.log 2>&1 &

env_name="Hopper-v2"
CUDA_VISIBLE_DEVICES=3 nohup python \
toolbox/action_distribution/train_mixture_gaussian.py \
--exp-name ${exp_name}_${env_name} --num-gpus 1 --env ${env_name} \
--num-seeds 1 > ${exp_name}_${env_name}.log 2>&1 &

env_name="HalfCheetah-v2"
nohup python \
toolbox/action_distribution/train_mixture_gaussian.py \
--exp-name ${exp_name}_${env_name} --num-gpus 4 --env ${env_name} \
--num-seeds 1 > ${exp_name}_${env_name}.log 2>&1 &

env_name="Humanoid-v2"
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python \
toolbox/action_distribution/train_mixture_gaussian.py \
--exp-name ${exp_name}_${env_name} --num-gpus 4 --env ${env_name} \
--num-seeds 1 > ${exp_name}_${env_name}.log 2>&1 &
