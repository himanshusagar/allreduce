import argparse
from math import floor

import torch
import logging
import time
from torch import distributed as dist
from utils import *

SEND_OK = True
RECV_OK = True
DEBUG = False

class RecursiveAllReduce:
    def __init__(self):
        self.globalTensor = torch.zeros(TENSOR_SIZE)

    def init_process(self, master_ip, rank, world_size):
        dist.init_process_group(backend="gloo",
                                init_method="tcp://" + master_ip + ":6585",
                                rank=rank,
                                world_size=world_size)
        self.my_rank = dist.get_rank()
        self.globalTensor[self.my_rank] = 1

    def sendTensors(self , partner_rank, begin , end):
        my_section_tensor = section_tensor(self.globalTensor, begin  , end)
        dist.send(my_section_tensor, dst=partner_rank)

    def recvTensors(self, partner_rank, begin , end):
        partner_size = end - begin + 1
        partner_section_tensor = torch.zeros(partner_size)
        s = time.time()
        dist.recv(partner_section_tensor, src=partner_rank)
        e = time.time()
        self.globalTensor = perform_op_tensor(self.globalTensor, begin , end , partner_section_tensor)
        print("Finished send recv from ", partner_rank, " at b = ", begin , "end =" , end , "in ", e - s, " seconds ", self.globalTensor)

    def reduce_scatter(self , left,  right):
        if(left >= right):
            return
        size = right - left + 1
        mid  = floor( (left + right)/2 )
        partner_rank = partner_index(self.my_rank , mid , size)

        if (self.my_rank <= mid):
            self.sendTensors(partner_rank , mid + 1  , right)
            self.recvTensors(partner_rank , left , mid)
        else:
            self.recvTensors(partner_rank , left , mid)
            self.sendTensors(partner_rank , mid + 1 , right)

        if(self.my_rank <= mid):
            self.reduce_scatter(left , mid)
        else:
            self.reduce_scatter(mid + 1 , right)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)

    args = parser.parse_args()
    rec = RecursiveAllReduce()
    rec.init_process(master_ip=args.master_ip,
                 rank=args.rank,
                 world_size=args.num_nodes)
    rec.reduce_scatter( 0 , 15)
