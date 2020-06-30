import torch
import sys

def fmt_weights(name, tensor):
    print(f"const float {name}[] = {{", end='')
    for value in tensor.reshape(-1):
        print(f"{value},", end='')
    print("};")

if __name__ == '__main__':
    weights = torch.load(sys.argv[1], map_location='cpu')
    fmt_weights('EMBEDDING_WEIGHTS', weights['embedding.weight'])
