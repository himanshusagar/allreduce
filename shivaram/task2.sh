#!/bin/bash

# Assumes node0 is at IP 10.10.1.1
# Assumes worker nodes have hostname node1, node2, ... node15
# Assumes you can ssh from node0 to all the other nodes.

#python3 ring.py --master-ip 10.10.1.1 --num-nodes 16 --rank 0 

parallel-ssh -i -h ~/followers "cd allreduce && git pull"

KB=1024
k=10
T_SIZE=$((k*KB))
T_SIZE=$((T_SIZE*KB))
incr=1
for j in 16
do
	j_lim=$((j-incr))
	for i in `seq 1 $j_lim`
	do
		RANK=$i
        	ssh -f a$i "nohup python3 /users/hsagar/allreduce/shivaram/ring_total.py -t $T_SIZE --master-ip 10.10.1.1 -p 6005  --num-nodes $j --rank $RANK"
	done
	python3 ~/allreduce/shivaram/ring_total.py -t $T_SIZE --master-ip 10.10.1.1 -p 6005 --num-nodes $j --rank 0
done

