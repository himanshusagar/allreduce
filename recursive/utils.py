from math import floor

import torch

TENSOR_SIZE = 16
WORLD_SIZE = 16
SECTION_SIZE =  int( TENSOR_SIZE / WORLD_SIZE ) # 64


commDict = [
    {   0: 1, 1: 0,
        2: 3, 3: 2,
        4: 5, 5: 4,
        6: 7, 7: 6,
        8: 9, 9: 8,
        10: 11, 12: 13,
        14: 15, 15: 14,
    },
    {   0: 2, 2: 0,
        4: 6, 6: 4,
        8: 10, 10: 8,
        12: 14, 14: 12,
    },
    {   0: 4, 4: 0,
        8: 12, 12: 8,
    },
    {   0: 8, 8: 0,
    }
]

def partner_index(rank , mid , size):
    if(rank <= mid):
        return rank + floor(size / 2);
    else:
        return rank - floor(size / 2)
    # level = int(level)
    # rank = int(rank)
    # return commDict[level][rank]

def section_tensor(wholeTensor, begin , end):
    return wholeTensor[begin : end + 1];

def perform_op_tensor(wholeTensor, begin , end , section_tensor):
    index = begin
    for i in range( end - begin + 1):
        wholeTensor[index] += section_tensor[i];
        index += 1
    return wholeTensor;

def clearNonPortion(rank , opTensor):
    begin = rank * SECTION_SIZE
    end = begin + SECTION_SIZE - 1
    print("begin and end " ,  begin , end)
    for i in range(TENSOR_SIZE):
        if(begin <= i and i <= end ):
            opTensor[i] = opTensor[i]
        else:
            opTensor[i] = 0
    return opTensor

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
