import torch
import sys

from model import NODE_TYPES, CHANNELS, MODULES, HIDDEN
BN_EPS = 1e-5

def fmt_weights(tensor):
    for value in tensor.view(-1):
        print(f"\t{value},")

if __name__ == '__main__':
    weights = torch.load(sys.argv[1], map_location='cpu')

    print(f"#define NODE_TYPES {NODE_TYPES}")
    print(f"#define CHANNELS {CHANNELS}")
    print(f"#define BN_EPS {BN_EPS}")
    print(f"#define MODULES {MODULES}")
    print(f"#define HIDDEN {HIDDEN}")
    print("static const float EMBED_WEIGHTS_DATA[] = {")
    fmt_weights(weights['embedding.weight'])
    print("};")

    print("static const float OUT_WEIGHTS_DATA[] = {")
    fmt_weights(weights['conv.out.weight'])
    for i in range(4):
        fmt_weights(weights[f'res.{i}.conv1.out.weight'])
        fmt_weights(weights[f'res.{i}.conv2.out.weight'])
    print("};")

    print("static const float BACK_WEIGHTS_DATA[] = {")
    fmt_weights(weights['conv.back.weight'])
    for i in range(4):
        fmt_weights(weights[f'res.{i}.conv1.back.weight'])
        fmt_weights(weights[f'res.{i}.conv2.back.weight'])
    print("};")

    print("static const float BN_MEAN_WEIGHTS_DATA[] = {")
    for i in range(4):
        fmt_weights(weights[f'res.{i}.bn1.running_mean'])
        fmt_weights(weights[f'res.{i}.bn2.running_mean'])
    print("};")

    print("static const float BN_VAR_WEIGHTS_DATA[] = {")
    for i in range(4):
        fmt_weights(weights[f'res.{i}.bn1.running_var'])
        fmt_weights(weights[f'res.{i}.bn2.running_var'])
    print("};")
