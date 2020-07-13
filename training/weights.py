import torch
import sys

from model import NODE_TYPES, CHANNELS, HIDDEN, LAYERS
BN_EPS = 1e-5

def fmt_weights(tensor):
    for value in tensor.reshape(-1):
        print(f"\t{value},")

if __name__ == '__main__':
    weights = torch.load(sys.argv[1], map_location='cpu')

    print(f"#define NODE_TYPES {NODE_TYPES}")
    print(f"#define CHANNELS {CHANNELS}")
    print(f"#define HIDDEN {HIDDEN}")
    print(f"#define LAYERS {LAYERS}")
    output_bias = float(weights['output.bias'])
    print(f"#define OUTPUT_BIAS {output_bias}")

    print("static const float EMBED_WEIGHTS_DATA[] = {")
    fmt_weights(weights['embedding.weight'])
    print("};")

    print("static const float OUT_WEIGHTS_DATA[] = {")
    for i in range(LAYERS):
        fmt_weights(weights[f'conv.{i}.out.weight.weight'].t())
    print("};")

    print("static const float BACK_WEIGHTS_DATA[] = {")
    for i in range(LAYERS):
        fmt_weights(weights[f'conv.{i}.back.weight.weight'].t())
    print("};")

    print("static const float HIDDEN_WEIGHT_DATA[] = {")
    fmt_weights(weights['hidden.weight'].t())
    print("};")
    print("static const float HIDDEN_BIAS_DATA[] = {")
    fmt_weights(weights['hidden.bias'])
    print("};")

    print("static const float OUTPUT_WEIGHT_DATA[] = {")
    fmt_weights(weights['output.weight'].t())
    print("};")
