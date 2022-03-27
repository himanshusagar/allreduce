import argparse
import torch
import logging
import time
from torch import distributed as dist
from recursive.utils import *

SEND_OK = True;
RECV_OK = True;
DEBUG = False;

def init_process(master_ip, rank, world_size):
    dist.init_process_group(backend="gloo",
                            init_method="tcp://" + master_ip + ":6585",
                            rank=rank,
                            world_size=world_size)

def shouldSend(my_rank):
    return my_rank < partner_index(my_rank)

def main():
    # Create a random tensor
    globalTensor = torch.zeros(TENSOR_SIZE)

    # Each machine will have one section.
    my_rank = dist.get_rank()
    if DEBUG:
        globalTensor[my_rank] = 1;

    for level in range(4):
        partner_rank = partner_index(level, my_rank)

        if(SEND_OK):
            my_section_tensor = section_tensor(globalTensor, my_rank)
            dist.send(my_section_tensor, dst=partner_rank)

        if(RECV_OK):
            partner_section_tensor = torch.zeros(SECTION_SIZE)
            s = time.time()
            dist.recv(partner_section_tensor, src=partner_rank)
            e = time.time()
            globalTensor = perform_op_tensor(globalTensor , partner_rank , partner_section_tensor)

        print("Finished send recv from ", partner_rank, " at level" ,  level ,  "in ", e - s, " seconds " , globalTensor)

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
