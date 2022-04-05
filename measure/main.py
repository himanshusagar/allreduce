import argparse
import torch
import logging
import time

from torch import distributed as dist

DEVICE = "cpu"
TENSOR_SIZE = 1024
import numpy as np

def init_process(master_ip, rank, world_size):
    dist.init_process_group(backend="gloo",
                            init_method="tcp://" + master_ip + ":9876",
                            rank=rank,
                            world_size=world_size)

def measure(timings):
    print("About to measure Alpha and Beta")
    b = np.array([ timings[1] ])
    A = np.array([np.array(1, TENSOR_SIZE * 1)])

    for i in range(2, dist.get_world_size()):
        A = np.append( A , np.array(1, TENSOR_SIZE * i), axis = 0)
    for i in range(2 , dist.get_world_size()):
        np.append(b , np.array(timings[0]) )

    print(np.shape(A) , np.shape(b))
    print(A)
    print(b)
    print( np.linalg.solve(A, b) )


def main():
    # Create a random tensor
    t = torch.rand(TENSOR_SIZE * dist.get_rank())
    # Send the tensor to rank 0
    timings = [0]
    if dist.get_rank() == 0:
        # Recv tensors from all ranks in an array
        recv_buffers = [torch.zeros(TENSOR_SIZE * dist.get_rank()) for i in range(1, dist.get_world_size())]
        for i in range(1, dist.get_world_size()):
            s = time.time()
            dist.recv(recv_buffers[i-1], src=i)
            e = time.time()
            print("Finished recv from ", i, " in ", e-s, " seconds")
            timings.append = (e - s)
        measure(timings)
    else:
        dist.send(t, dst=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)

    args = parser.parse_args()
    init_process(master_ip=args.master_ip,
                 rank=args.rank,
                 world_size=args.num_nodes)
    main()
