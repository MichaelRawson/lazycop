import sys

from model import Model
from data import examples

def fmt_tensor(tensor):
    for value in tensor.reshape(-1):
        print(f"\t{value},")

if __name__ == '__main__':
    data = examples('data.gz', 64)
    batch = next(data)
    nodes = batch['nodes']
    sources = batch['sources']
    targets = batch['targets']
    num_graphs = batch['graph_count']
    node_batch = batch['node_boundary']
    edge_batch = batch['edge_boundary']

    print(f"const uint32_t EXAMPLE_NUM_NODES = {len(nodes)};")
    print(f"const uint32_t EXAMPLE_NUM_EDGES = {len(sources)};")
    print(f"const uint32_t EXAMPLE_NUM_GRAPHS = {num_graphs};")
    print("const uint32_t EXAMPLE_NODES[] = {")
    fmt_tensor(nodes)
    print("};")
    print("const uint32_t EXAMPLE_SOURCES[] = {")
    fmt_tensor(sources)
    print("};")
    print("const uint32_t EXAMPLE_TARGETS[] = {")
    fmt_tensor(targets)
    print("};")
    print("const uint32_t EXAMPLE_NODE_BATCH[] = {")
    fmt_tensor(node_batch)
    print("};")
    print("const uint32_t EXAMPLE_EDGE_BATCH[] = {")
    fmt_tensor(edge_batch)
    print("};")
