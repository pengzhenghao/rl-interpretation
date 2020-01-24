#!/bin/bash

#SBATCH --job-name=0117-dece-cluster

#SBATCH --mem-per-cpu=0
#SBATCH --nodes=1
#SBATCH --tasks-per-node 1
#SBATCH --gres=gpu:K80:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pengzh@ie.cuhk.edu.hk
#SBATCH -o 0117-dece-cluster.log
#SBATCH --exlusive


worker_num=1 # num node - 1
cd ~/novel-rl/scripts
# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

echo "Current nodes:"
echo $nodes

node1=${nodes_array[0]}

ips=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address

# In some case the ips is a two element array
# ip_array=( $ips )
# ip_prefix=${ip_array[1]} # The first element is ipv6 address

# But in some case it's not
ip_prefix=$ips

suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py
echo "ip_head"
echo $ip_head

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5

python -u 0117-dece-cluster.py $redis_password 3 # Pass the total number of allocated CPUs
