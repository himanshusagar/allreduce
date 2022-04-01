#!/bin/bash

# Assumes node0 is at IP 10.10.1.1
# Assumes worker nodes have hostname node1, node2, ... node15
# Assumes you can ssh from node0 to all the other nodes.

parallel-ssh -i -h ~/followers "cd dev && cd allreduce && git pull"
T_SIZE=3
W_SIZE=3

python3 ~/dev/allreduce/recursive/main.py --master-ip 10.10.1.1 --num-nodes $W_SIZE --rank 0 --tensor-size $T_SIZE &

for i in `seq 1 $W_SIZE`
do
        RANK=$i
        echo "Staring rank $RANK"
        ssh -f a$i "nohup python3 ~/dev/allreduce/recursive/main.py --master-ip 10.10.1.1 --num-nodes $W_SIZE --rank $RANK --tensor-size $T_SIZE"
done

#python3 main.py --master-ip 10.10.1.1 --num-nodes 16 --rank 0