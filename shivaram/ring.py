import argparse
import torch
import logging
import time

from torch import distributed as dist

DEVICE = "cpu"
TENSOR_SIZE = 16

def init_process(master_ip, rank, world_size):
    dist.init_process_group(backend="gloo",
                            init_method="tcp://" + master_ip + ":6590",
                            rank=rank,
                            world_size=world_size)

def ring_gather(t, comm_size, world_size, me, prev, _next_):
    curi = me
    for i in range(0, world_size):
        if(me%2 == 0):
            send_buf = torch.zeros(comm_size)
            for idx in range(0,comm_size):
                send_buf[idx]=t[curi*comm_size + idx]
            s=time.time()
            dist.send(send_buf, dst=_next_)
            e=time.time()
            print("Finished send to ", _next_, " in ", e - s, " seconds ")

            recv_buf = torch.zeros(comm_size)
            s=time.time()
            dist.recv(recv_buf, src=prev)
            e=time.time()
#            print("Finished recv from ", prev, " in ", e - s, " seconds ")
            k = 0
            curi = curi-1
            if(curi < 0):
                curi = world_size-1
            for j in range(curi*comm_size, (curi+1)*comm_size):
                t[j] = t[j]+recv_buf[k]
                k=k+1
        else:
            recv_buf = torch.zeros(comm_size)
            s=time.time()
            dist.recv(recv_buf, src=prev)
            e=time.time()
#            print("Finished recv from ", prev, " in ", e - s, " seconds ")
            k = 0
            curi = curi-1
            if(curi < 0):
                curi = world_size-1
            for j in range(curi*comm_size, (curi+1)*comm_size):
                t[j] = t[j]+recv_buf[k]
                k=k+1

            curi = curi+1
            if (curi >= world_size):
                curi = 0
            send_buf = torch.zeros(comm_size)
            for idx in range(0,comm_size):
                send_buf[idx]=t[curi*comm_size + idx]
            s=time.time()
            dist.send(send_buf, dst=_next_)
            e=time.time()
            print("Finished send to ", _next_, " in ", e - s, " seconds ")
            curi = curi-1
            if(curi < 0):
                curi = world_size-1


def ring_scatter(t, comm_size, world_size, me, prev, _next_):
    curi = me+1
    if(curi >= world_size):
        curi = 0;
    for i in range(0, world_size):
        if(me%2 == 0):
            send_buf = torch.zeros(comm_size)
            for idx in range(0,comm_size):
                send_buf[idx]=t[curi*comm_size + idx]
            s=time.time()
            dist.send(send_buf, dst=_next_)
            e=time.time()
            print("Finished send to ", _next_, " in ", e - s, " seconds ")

            recv_buf = torch.zeros(comm_size)
            s=time.time()
            dist.recv(recv_buf, src=prev)
            e=time.time()
#            print("Finished recv from ", prev, " in ", e - s, " seconds ")
            k = 0
            curi = curi-1
            if(curi < 0):
                curi = world_size-1
            for j in range(curi*comm_size, (curi+1)*comm_size):
                t[j] = recv_buf[k]
                k=k+1
        else:
            recv_buf = torch.zeros(comm_size)
            s=time.time()
            dist.recv(recv_buf, src=prev)
            e=time.time()
#            print("Finished recv from ", prev, " in ", e - s, " seconds ")
            curi = curi-1
            if(curi < 0):
                curi = world_size-1
            k = 0
            for j in range(curi*comm_size, (curi+1)*comm_size):
                t[j] = recv_buf[k]
                k=k+1

            curi = curi+1
            if (curi >= world_size):
                curi = 0
            send_buf = torch.zeros(comm_size)
            for idx in range(0,comm_size):
                send_buf[idx]=t[curi*comm_size + idx]
            s=time.time()
            dist.send(send_buf, dst=_next_)
            e=time.time()
            print("Finished send to ", _next_, " in ", e - s, " seconds ")
            curi = curi-1
            if(curi < 0):
                curi = world_size-1


def main(tensor_size):
    # Create a random tensor
    t = torch.rand(tensor_size)
    for i in range(0,tensor_size):
        t[i]=i
    # indices to send and receive from
    me = dist.get_rank()
    world_size = dist.get_world_size()
    comm_size = int(tensor_size/world_size)

    prev = me-1
    if(prev < 0):
        prev = world_size - 1
    _next_ = me+1
    if(_next_ >= world_size):
        _next_ = 0
    
    ring_gather(t, comm_size, world_size, me, prev, _next_)
    ring_scatter(t, comm_size, world_size, me, prev, _next_)
    print(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-size", "-t", required=False, type=int, default=16)
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=False, type=int, default=16)
    parser.add_argument("--rank", "-r", required=True, type=int)

    args = parser.parse_args()
    init_process(master_ip=args.master_ip,
                 rank=args.rank,
                 world_size=args.num_nodes)
    main(tensor_size=args.tensor_size)
