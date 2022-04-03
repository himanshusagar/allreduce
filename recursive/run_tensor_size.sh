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

#echo "inside run_func $T_SIZE $W_SIZE $END_LOOP $PORT_VAL"

for i in `seq 1 $END_LOOP`
do
        RANK=$i
        #echo "Staring rank $RANK $WORLD_SIZE $T_SIZE"
        ssh -f a$i "nohup python3 ~/dev/allreduce/recursive/main.py --master-ip 10.10.1.1 --num-nodes $W_SIZE --rank $RANK --tensor-size $T_SIZE --port $PORT_VAL"
done

python3 ~/dev/allreduce/recursive/main.py --master-ip 10.10.1.1 --num-nodes $W_SIZE --rank 0 --tensor-size $T_SIZE --port $PORT_VAL &

}

echo "Start of script..."
KB=1024
STEPSIZE=$((KB*512))
STEPSIZE=$((STEPSIZE-1))

MB=$((KB*KB))
TenMB=$(($MB*10))
HundredMB=$(($MB*100))


WORLD_SIZE=16
PORT_VAL=50000

for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072; do
      tensor_size=$(( $KB * $i))
      #echo "$tensor_size $i"
      run_func $tensor_size $WORLD_SIZE $PORT_VAL
      PORT_VAL=$((PORT_VAL+2))
done


#PORT_VAL=6581
#
#for i in 1 2 4 8 16 32 128 256 512; do
#      tensor_size=$(( $KB * $i))
#      #echo "$tensor_size $i"
#      run_func $tensor_size $WORLD_SIZE $PORT_VAL
#      PORT_VAL=$((PORT_VAL+2))
#done

#
#for i in 1 10 20 40 60 80 100 ; do
#      tensor_size=$(( $MB * $i))
#      #echo "$tensor_size $i"
#      run_func $tensor_size $WORLD_SIZE $PORT_VAL
#      PORT_VAL=$((PORT_VAL+2))
#done
