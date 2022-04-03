import argparse
import torch
import logging
import time

from torch import distributed as dist

DEVICE = "cpu"

def init_process(master_ip, port, rank, world_size):
    dist.init_process_group(backend="gloo",
                            init_method="tcp://" + master_ip + ":" + port,
                            rank=rank,
                            world_size=world_size)

def ring_gather(t, comm_size, world_size, me, prev, _next_):
    curi = me
    for i in range(0, world_size-1):
        if(me == 0):
            send_buf = torch.zeros(comm_size)
            for idx in range(0,comm_size):
                send_buf[idx]=t[curi*comm_size + idx]
            dist.send(send_buf, dst=_next_)

            recv_buf = torch.zeros(comm_size)
            dist.recv(recv_buf, src=prev)
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
            dist.send(send_buf, dst=_next_)
#            print("Finished send to ", _next_, " in ", e - s, " seconds ")
            curi = curi-1
            if(curi < 0):
                curi = world_size-1

def ring_scatter(t, comm_size, world_size, me, prev, _next_):
    curi = me+1
    if(curi >= world_size):
        curi = 0;
    for i in range(0, world_size-1):
        if(me == 0):
            send_buf = torch.zeros(comm_size)
            for idx in range(0,comm_size):
                send_buf[idx]=t[curi*comm_size + idx]
            dist.send(send_buf, dst=_next_)

            recv_buf = torch.zeros(comm_size)
            dist.recv(recv_buf, src=prev)
            k = 0
            curi = curi-1
            if(curi < 0):
                curi = world_size-1
            for j in range(curi*comm_size, (curi+1)*comm_size):
                t[j] = recv_buf[k]
                k=k+1
        else:
            recv_buf = torch.zeros(comm_size)
            dist.recv(recv_buf, src=prev)
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
            dist.send(send_buf, dst=_next_)
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

    s=time.time()
    ring_gather(t, comm_size, world_size, me, prev, _next_)
    ring_scatter(t, comm_size, world_size, me, prev, _next_)
    e=time.time()

    times_buf = torch.zeros(1)
    times_buf[0] = e-s
    if(me != 0):
        dist.send(times_buf, dst=0)
    else:
        all_times_buf = torch.zeros(1,world_size)
        all_times_buf[0][0] = times_buf[0]
        for i  in range(1,world_size):
            dist.recv(all_times_buf[0][i], src=i)

        print("Tensor size : ", tensor_size, " World size(average over this) : " , world_size, " , Total AllReduce time across all VMs : ", all_times_buf.sum().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-size", "-t", required=False, type=int, default=16)
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--port", "-p", required=False, type=str, default="6003")
    parser.add_argument("--num-nodes", "-n", required=False, type=int, default=16)
    parser.add_argument("--rank", "-r", required=True, type=int)

    args = parser.parse_args()
    init_process(master_ip=args.master_ip, port=args.port,
                 rank=args.rank,
                 world_size=args.num_nodes)
#    print(args.port)
#    print(args.num_nodes)
#    print(args.tensor_size)
    main(tensor_size=args.tensor_size)
