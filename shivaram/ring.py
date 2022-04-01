import argparse
import torch
import logging
import time

from torch import distributed as dist

DEVICE = "cpu"
TENSOR_SIZE = 1024

def init_process(master_ip, rank, world_size):
    dist.init_process_group(backend="gloo",
                            init_method="tcp://" + master_ip + ":6585",
                            rank=rank,
                            world_size=world_size)


def main():
    # Create a random tensor
    t = torch.rand(TENSOR_SIZE)
    # indices to send and receive from
    me = dist.get_rank()
    world_size = dist.get_world_size()
    print(world_size)
    comm_size = int(TENSOR_SIZE/world_size)

    prev = me-1
    if(prev < 0):
        prev = world_size - 1
    _next_ = me+1
    if(_next_ >= world_size):
        _next_ = 0

    curi = me
    for i in range(0, world_size):
        if(i%2 == 0):
            send_buf = torch.zeros(comm_size)
            for idx in range(0,comm_size):
                send_buf[idx]=t[curi*comm_size + idx]
            dist.send(send_buf, dst=_next_)

            recv_buf = torch.zeros(comm_size)
            dist.recv(recv_buf, src=prev)
            print("Finished recv from ", prev)
            k = 0
            curi = curi-1
            if(curi < 0):
                curi = world_size-1
            for j in range(curi*comm_size, (curi+1)*comm_size):
                t[j] = t[j]+recv_buf[k]
                k=k+1
        else:
            recv_buf = torch.zeros(comm_size)
            dist.recv(recv_buf, src=prev)
            print("Finished recv from ", prev)
            k = 0
            curi = curi-1
            if(curi < 0):
                curi = world_size-1
            for j in range(curi*comm_size, (curi+1)*comm_size):
                t[j] = t[j]+recv_buf[k]
                k=k+1

            curi = me
            send_buf = torch.zeros(comm_size)
            for idx in range(0,comm_size):
                send_buf[idx]=t[curi*comm_size + idx]
            dist.send(send_buf, dst=_next_)




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
