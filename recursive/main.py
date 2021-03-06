import argparse
from math import floor

import torch
import logging
import time
from torch import distributed as dist
from base import BaseClass

PORT_VAL = 5580;
DEBUG = False

class RecursiveAllReduce(BaseClass):
    def __init__(self, tensor_size, world_size, master_ip , rank):
        super().__init__(tensor_size, world_size)
        self.globalTensor = torch.rand(self.TENSOR_SIZE)
        self.init_process(master_ip , rank, world_size);


    def init_process(self, master_ip, rank, world_size):
        dist.init_process_group(backend="gloo",
                                init_method="tcp://" + master_ip + ":" + str(PORT_VAL),
                                rank=rank,
                                world_size=world_size)
        self.my_rank = dist.get_rank()
        for i in range(self.TENSOR_SIZE):
            self.globalTensor[i] = i
        if(DEBUG):
            print("Initial Tensor " , self.my_rank , self.globalTensor)

    def clearNonPortion(self):
        begin = self.my_rank * self.SECTION_SIZE
        end = begin + self.SECTION_SIZE - 1
        for i in range(self.TENSOR_SIZE):
            if begin <= i <= end:
                self.globalTensor[i] = self.globalTensor[i]
            else:
                self.globalTensor[i] = 0

    def sendTensors(self , partner_rank, begin , end):
        #my_section_tensor = self.section_tensor(begin  , end)
        #s = time.time()
        dist.send(self.globalTensor[begin: end + 1], dst=partner_rank)
        #e = time.time()
        #self.send_time.append(e - s);


    def recvTensors(self, partner_rank, begin , end, assign):
        partner_size = end - begin + 1
        partner_section_tensor = torch.zeros(partner_size)
        #s = time.time()
        dist.recv(partner_section_tensor, src=partner_rank)
        #e = time.time()
        #self.recv_time.append(e - s)
        if (assign):
            self.globalTensor[begin:end + 1] = partner_section_tensor
        else:
            self.globalTensor[begin:end + 1] += partner_section_tensor

        #if (DEBUG):
        #    print("Finished send recv from ", partner_rank, " at b = ", begin , "end =" , end , "in ", e - s, " seconds ", self.globalTensor)

    def reduce_scatter(self , left,  right):
        if(left >= right):
            return
        size = right - left + 1
        mid  = floor( (left + right)/2 )
        partner_rank = self.partner_index(self.my_rank , mid , size)

        if (self.my_rank <= mid):
            self.sendTensors(partner_rank , mid + 1  , right)
            self.recvTensors(partner_rank , left , mid, False)
        else:
            self.recvTensors(partner_rank ,  mid + 1 , right, False)
            self.sendTensors(partner_rank , left , mid)

        if(self.my_rank <= mid):
            self.reduce_scatter(left , mid)
        else:
            self.reduce_scatter(mid + 1 , right)

    def all_gather(self , left, right):
        if (left >= right):
            return
        size = right - left + 1
        mid = floor((left + right) / 2)
        partner_rank = self.partner_index(self.my_rank, mid, size)

        if(self.my_rank <= mid):
            self.all_gather(left , mid)
        else:
            self.all_gather(mid + 1 , right)

        if(self.my_rank <= mid):
            self.sendTensors(partner_rank, left, mid)
            self.recvTensors(partner_rank, mid + 1, right, True)
        else:
            self.recvTensors(partner_rank, left, mid, True)
            self.sendTensors(partner_rank , mid + 1  , right)


    def accumulate(self):
        t = torch.zeros(1)
        if dist.get_rank() == 0:
            tmp_list = self.get_tmp_list()
            recv_buffers = [torch.zeros(1) for i in range(0, dist.get_world_size())]
            recv_buffers[0] = self.take_sum(tmp_list)
            for i in range(1, dist.get_world_size()):
                s = time.time()
                dist.recv(recv_buffers[i], src=i)
                e = time.time()
            #print("Finished recv in total ", recv_buffers);
            toPrint = ""
            toPrint += str(self.TENSOR_SIZE) + "," + str(self.WORLD_SIZE) + "," + str(self.take_sum(recv_buffers));
            print( toPrint )
        else:
            tmp_list = self.get_tmp_list()
            t[0] = self.take_sum(tmp_list);
            dist.send(t, dst=0)


    def algo(self):
        s = time.time();
        self.reduce_scatter(0, self.WORLD_SIZE - 1)
        e = time.time();
        self.tot_time.append(e - s);
        #self.clearNonPortion()
        #backup = torch.clone(self.globalTensor)
        s = time.time()
        self.all_gather(0, self.WORLD_SIZE - 1)
        e = time.time()
        self.tot_time.append(e - s);
        #if(DEBUG):
        #    print("End Tensor ", rec.my_rank, " after reduce_scatter", backup, " and all_gather ", rec.globalTensor)
        self.accumulate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-ip", "-m", required=True, type=str)
    parser.add_argument("--num-nodes", "-n", required=True, type=int)
    parser.add_argument("--tensor-size", "-t", required=True, type=int)
    parser.add_argument("--rank", "-r", required=True, type=int)
    parser.add_argument("--port", "-p", required=True, type=int)

    args = parser.parse_args()
    if(args.port > 0):
        PORT_VAL = args.port
    rec = RecursiveAllReduce(args.tensor_size , args.num_nodes , args.master_ip , args.rank)
    rec.algo()


