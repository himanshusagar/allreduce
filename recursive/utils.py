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

def partner_index(level, rank):
    level = int(level)
    rank = int(rank)
    return commDict[level][rank]

def section_tensor(wholeTensor, rank):
    begin = int(rank * SECTION_SIZE);
    end = int(begin + SECTION_SIZE);
    return wholeTensor[begin : end];

def perform_op_tensor(wholeTensor, rank, section_tensor):
    index = rank * SECTION_SIZE
    for i in range(SECTION_SIZE):
        wholeTensor[index] += section_tensor[i];
        index += 1
    return wholeTensor;


def shouldSendFirst( level , my_rank):
    return my_rank < partner_index(level , my_rank)

if __name__ == '__main__':
    globalTensor = torch.zeros(TENSOR_SIZE)

    for i in range(TENSOR_SIZE):
        globalTensor[i] = i;

    print("Global Tensor")
    print(globalTensor)
    rank = 2
    print(partner_index(1 , rank))
    print(section_tensor(globalTensor , rank))

    partner_section = section_tensor(globalTensor, 3);
    print("partner_section Tensor")
    print(partner_section)
    globalTensor = perform_op_tensor(globalTensor , 3 , partner_section);
    print("globalTensor" , globalTensor);
