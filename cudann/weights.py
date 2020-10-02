import torch
import sys

from model import NODE_TYPES, CHANNELS, HIDDEN, LAYERS, Model

def fmt_weights(tensor):
    for value in tensor.reshape(-1):
        print(f"\t{value},")

if __name__ == '__main__':
    model = Model()
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()
    model.fuse()
    weights = model.state_dict()

    print(f"const uint32_t NODE_TYPES = {NODE_TYPES};")
    print(f"const uint32_t CHANNELS = {CHANNELS};")
    print(f"const uint32_t HIDDEN = {HIDDEN};")
    print(f"const uint32_t LAYERS = {LAYERS};")

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

    print("static const float OUT_BIAS_DATA[] = {")
    for i in range(LAYERS):
        fmt_weights(weights[f'conv.{i}.out.weight.bias'])
    print("};")

    print("static const float BACK_BIAS_DATA[] = {")
    for i in range(LAYERS):
        fmt_weights(weights[f'conv.{i}.back.weight.bias'])
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
