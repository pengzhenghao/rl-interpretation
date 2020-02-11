#!/bin/bash

#SBATCH --job-name=test

#SBATCH --mem-per-cpu=0
#SBATCH --nodes=4
#SBATCH --tasks-per-node 1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

num_gpus=4
worker_num=3 # num node - 1


# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
# nodes222=$(scontrol show hostnames) # Getting the node names
nodes_array=( $nodes )

echo "Current nodes:"
echo $nodes

node1=${nodes_array[0]}

ips=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
ip_array=( $ips )
ip_prefix=${ip_array[1]} # The first element is ipv6 address


suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py
echo "ip_head"
echo $ip_head

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password --num-gpus=$num_gpus & # Starting the head
sleep 5

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password --num-gpus=$num_gpus & # Starting the workers
  sleep 5
done

# for (( i=$cpu_worker_start; i<=$cpu_worker_index; i++ ))
# do
#   node3=${nodes_array[$i]}
#   srun --nodes=1 --ntasks=1 --pack-group=1 -w $node3 ray start --block --address=$ip_head --redis-password=$redis_password &
#   sleep 5
# done

python -u 0116-dece-cluster.py $redis_password 3 # Pass the total number of allocated CPUs
