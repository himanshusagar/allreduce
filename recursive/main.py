import argparse
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

    def sendTensors(self , partner_rank):
        my_section_tensor = section_tensor(self.globalTensor, self.my_rank)
        dist.send(my_section_tensor, dst=partner_rank)

    def recvTensors(self, level):
        partner_rank = partner_index(level , rank)
        partner_section_tensor = torch.zeros(SECTION_SIZE)
        s = time.time()
        dist.recv(partner_section_tensor, src=partner_rank)
        e = time.time()
        self.globalTensor = perform_op_tensor(self.globalTensor, partner_rank, partner_section_tensor)
        print("Finished send recv from ", partner_rank, " at level", level, "in ", e - s, " seconds ", globalTensor)

    def main(self):
        for level in range(4):
            partner_rank = partner_index(level, self.my_rank)
            if(shouldSendFirst(self.my_rank)):
                self.sendTensors(partner_rank)
                self.recvTensors(level)
            else:
                self.recvTensors(level)
                self.sendTensors(partner_rank)

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
    rec.main()
