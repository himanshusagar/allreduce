import argparse
import torch
import logging
import time

from torch import distributed as dist

DEVICE = "cpu"
TENSOR_SIZE = int( (1024 * 1024 * 1024)/10 )
import numpy as np

def init_process(master_ip, rank, world_size):
    dist.init_process_group(backend="gloo",
                            init_method="tcp://" + master_ip + ":9786",
                            rank=rank,
                            world_size=world_size)

def measure(timings):
    print("About to measure Alpha and Beta")

    begin = 5
    end = 10

    b = np.array([ timings[begin] ])
    A = np.array([np.array( [1, TENSOR_SIZE * begin ] )])

    #for i in range(end, end+1):
    entry = np.array([np.array([1, TENSOR_SIZE * end])])
    A = np.append( A , entry , axis = 0)

    #for i in range(end , end+1):
    entry = np.array([timings[end]])
    b = np.append(b , entry )

    print(np.shape(A) , np.shape(b))
    print(A)
    print(b)
    print("HELLO")
    print( np.linalg.solve(A, b) )


def main():
    # Create a random tensor
    t = torch.rand(TENSOR_SIZE * dist.get_rank())
    # Send the tensor to rank 0
    timings = [0]
    if dist.get_rank() == 0:
        # Recv tensors from all ranks in an array
        recv_buffers = [torch.zeros(TENSOR_SIZE * i) for i in range(1, dist.get_world_size())]
        for i in range(1, dist.get_world_size()):
            s = time.time()
            dist.recv(recv_buffers[i-1], src=i)
            e = time.time()
            print("Finished recv from ", i, " in ", e-s, " seconds")
            timings.append(e - s)
        measure(timings)
    else:
        dist.send(t, dst=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)

    args = parser.parse_args()
    s = time.time()
    init_process(master_ip=args.master_ip,
                 rank=args.rank,
                 world_size=args.num_nodes)
    e = time.time()
    print("Finished init_process from ", dist.get_rank() , " in ", e - s, " seconds")
    main()
