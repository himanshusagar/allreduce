from math import floor
import numpy as np

import torch

class BaseClass:

    def __init__(self, tensor_size, world_size):
        self.TENSOR_SIZE = tensor_size
        self.WORLD_SIZE = world_size
        self.SECTION_SIZE = int(self.TENSOR_SIZE / self.WORLD_SIZE)  # 64
        self.send_time = []
        self.recv_time = []
        self.tot_time = []

    def calc(self , tmp_list):
        return np.mean(tmp_list);

    def get_tmp_list(self):
        #tmp_list = self.send_time;
        # tmp_list.extend(self.recv_time)
        return self.tot_time

    def partner_index(self, rank, mid, size):
        if (rank <= mid):
            return rank + floor(size / 2);
        else:
            return rank - floor(size / 2)

    def section_tensor(self, wholeTensor, begin, end):
        return wholeTensor[begin: end + 1];

    def perform_op_tensor(self, wholeTensor, begin, end, section_tensor):
        index = begin
        for i in range(end - begin + 1):
            wholeTensor[index] += section_tensor[i];
            index += 1
        return wholeTensor;

    # def clearNonPortion(self, rank, opTensor):
    #     begin = rank * self.SECTION_SIZE
    #     end = begin + self.SECTION_SIZE - 1
    #     for i in range(self.TENSOR_SIZE):
    #         if (begin <= i and i <= end):
    #             opTensor[i] = opTensor[i]
    #         else:
    #             opTensor[i] = 0
    #     return opTensor


if __name__ == '__main__':

    globalTensor = torch.zeros(TENSOR_SIZE)

    for i in range(TENSOR_SIZE):
        globalTensor[i] = i;

    print("Global Tensor")
    print(globalTensor)
    rank = 2
    print(partner_index(1 , rank , 5))
    begin = 3;
    end = 5;

    partner_section = section_tensor(globalTensor, begin , end)
    print("partner_section Tensor")
    print(partner_section)
    globalTensor = perform_op_tensor(globalTensor , begin , end , partner_section);
    print("globalTensor" , globalTensor);

    globalTensor = clearNonPortion(5 , globalTensor)

    print("globalTensor" , globalTensor);



