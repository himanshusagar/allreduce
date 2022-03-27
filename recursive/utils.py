import torch

TENSOR_SIZE = 16
WORLD_SIZE = 16
SECTION_SIZE =  TENSOR_SIZE / WORLD_SIZE # 64


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
    return commDict[level][rank]

def section_tensor(wholeTensor, rank):
    index = rank * SECTION_SIZE;
    return wholeTensor[index : index + SECTION_SIZE];

def perform_op_tensor(wholeTensor, rank, section_tensor):
    index = rank * SECTION_SIZE
    for i in range(SECTION_SIZE):
        wholeTensor[index] += section_tensor[i];
        index += 1
    return wholeTensor;


if __name__ == '__main__':
    globalTensor = torch.rand(TENSOR_SIZE)

    print("Global Tensor")
    print(globalTensor)
    rank = 2
    print(partner_index(1 , rank))
    print(section_tensor(globalTensor , rank))

    partner_section = section_tensor(globalTensor, 3);

    globalTensor = perform_op_tensor(globalTensor , rank , partner_section);
    print("globalTensor" , globalTensor);
