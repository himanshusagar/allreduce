#!/bin/bash

# Assumes node0 is at IP 10.10.1.1
# Assumes worker nodes have hostname node1, node2, ... node15
# Assumes you can ssh from node0 to all the other nodes.

parallel-ssh -i -h ~/followers "cd dev && cd allreduce && git pull"

run_func()
{

T_SIZE=$1
W_SIZE=$2
PORT_VAL=$3
END_LOOP=$(($W_SIZE-1))

#echo "inside node_count $T_SIZE $W_SIZE $END_LOOP $PORT_VAL"
for i in `seq 1 $END_LOOP`
do
        RANK=$i
        #echo "Staring rank $RANK $WORLD_SIZE $T_SIZE"
        ssh -f a$i "nohup python3 ~/dev/allreduce/recursive/main.py --master-ip 10.10.1.1 --num-nodes $W_SIZE --rank $RANK --tensor-size $T_SIZE --port $PORT_VAL"
done

python3 ~/dev/allreduce/recursive/main.py --master-ip 10.10.1.1 --num-nodes $W_SIZE --rank 0 --tensor-size $T_SIZE --port $PORT_VAL

}

echo "Start of script..."
KB=1024
MB=$((KB*KB))
TenMB=$(($MB*10))
PORT_VAL=9999
tensor_size=$TenMB

for WORLD_SIZE in 2 4 8 16; do
      run_func $tensor_size $WORLD_SIZE $PORT_VAL
done