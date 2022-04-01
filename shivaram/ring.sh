#!/bin/bash

# Assumes node0 is at IP 10.10.1.1
# Assumes worker nodes have hostname node1, node2, ... node15
# Assumes you can ssh from node0 to all the other nodes.

#python3 ring.py --master-ip 10.10.1.1 --num-nodes 16 --rank 0 

parallel-ssh -i -h ~/followers "cd allreduce && git pull"

python3 ~/allreduce/shivaram/ring.py --master-ip 10.10.1.1 --num-nodes 2 --rank 0 &

for i in `seq 1`
do
        RANK=$i
        echo "Starting rank $RANK"
        ssh -f a$i "nohup python3 /users/hsagar/allreduce/shivaram/ring.py --master-ip 10.10.1.1 --num-nodes 2 --rank $RANK"
done
