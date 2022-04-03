#!/bin/bash

# Assumes node0 is at IP 10.10.1.1
# Assumes worker nodes have hostname node1, node2, ... node15
# Assumes you can ssh from node0 to all the other nodes.

#python3 ring.py --master-ip 10.10.1.1 --num-nodes 16 --rank 0 

parallel-ssh -i -h ~/followers "cd allreduce && git pull"

KB=1024
PORT=6100
incr=1
#4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072
for j in 1024 2048 4096 8192 16384
do
	T_SIZE=$((j*KB))
	for i in `seq 1 15`
	do
		RANK=$i
        	ssh -f a$i "nohup python3 /users/hsagar/allreduce/shivaram/ring_opt.py -t $T_SIZE --master-ip 10.10.1.1 -p $PORT --num-nodes 16 --rank $RANK"
	done
	python3 ~/allreduce/shivaram/ring_opt.py -t $T_SIZE --master-ip 10.10.1.1 -p $PORT --num-nodes 16 --rank 0 &
	PORT=$((PORT+incr))

done

