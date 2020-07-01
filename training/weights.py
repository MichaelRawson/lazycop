import torch
import sys

def fmt_weights(tensor):
    for value in tensor.reshape(-1):
        print(f"\t{value},")

if __name__ == '__main__':
    weights = torch.load(sys.argv[1], map_location='cpu')

    print("const unsigned NODE_TYPES = 7;")
    print("const unsigned CHANNELS = 64;")
    print("const float EMBEDDING_WEIGHTS[7][64] = {")
    fmt_weights(weights['embedding.weight'].reshape(-1))
    print("};")

    print("const float OUT_WEIGHTS[9][CHANNELS][CHANNELS] = {")
    fmt_weights(weights['conv.out.weight'].t())
    for i in range(4):
        fmt_weights(weights[f'res.{i}.conv1.out.weight'].t())
        fmt_weights(weights[f'res.{i}.conv2.out.weight'].t())
    print("};")

    print("const float BACK_WEIGHTS[9][CHANNELS][CHANNELS] = {")
    fmt_weights(weights['conv.back.weight'].t())
    for i in range(4):
        fmt_weights(weights[f'res.{i}.conv1.back.weight'].t())
        fmt_weights(weights[f'res.{i}.conv2.back.weight'].t())
    print("};")
