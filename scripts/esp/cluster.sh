#!/bin/bash
# This file is used to test ray compatibility with sensetime cluster. Maintain: Peng Zhenghao

# ===== Allocate resources =====
#SBATCH --job-name=esp
#SBATCH --output=esp_atari_0413.log
#SBATCH --mem-per-cpu=0
#SBATCH --nodes=3
#SBATCH --cpus-per-task=64
#SBATCH --tasks-per-node 1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --partition=VI_SP_Y_1080TI

# ===== Define variables =====
worker_num=2
num_gpus=8  # per node
num_cpus=64 # per node

# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate
~/anaconda3/bin/activate dev

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
# nodes222=$(scontrol show hostnames) # Getting the node names
nodes_array=($nodes)

echo "Current nodes:"
echo $nodes

node1=${nodes_array[0]}

echo "Head Node:"
echo $node1

ips=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address

echo "ips: "
echo $ips

ip_array=($ips)
ip_prefix=${ip_array[0]} # The first element is ipv6 address

suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py
echo "ip_head"
echo $ip_head

# ===== Start the head node =====
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password --num-gpus=$num_gpus --num-cpus=$num_cpus &# Starting the head
sleep 20

# ===== Start worker node =====
for ((i = 1; i <= $worker_num; i++)); do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password --num-gpus=$num_gpus --num-cpus=$num_cpus &# Starting the workers
  sleep 20
done

# ===== Submit task =====

cd ~/novel-rl/

python -u toolbox/scripts/train_atari.py \
  --exp-name esp_atari_0413 \
  --redis-password $redis_password
