from ctypes import cast, c_uint, c_uint32, c_float, CDLL, POINTER
c_uint32p = POINTER(c_uint32)
c_floatp = POINTER(c_float)
import sys

import torch

from data import examples

so = CDLL('cuda/libmodel.so')
init = so.init
init.restype = None
init.argtypes = []
model = so.model
model.restype = None
model.argtypes = [
    c_uint32,
    c_uint32,
    c_uint32,
    c_uint32p,
    c_uint32p,
    c_uint32p,
    c_uint32p,
    c_floatp
]

if __name__ == '__main__':
    init()
    test = examples(sys.argv[1], self_loops=False)
    for nodes, sources, targets, rules, y in test:
        num_nodes = c_uint32(len(nodes))
        num_edges = c_uint32(len(sources))
        num_rules = c_uint32(len(rules))
        nodes = nodes.int()
        sources = sources.int()
        targets = targets.int()
        rules = rules.int()
        nodes_ptr = cast(nodes.data_ptr(), c_uint32p)
        sources_ptr = cast(sources.data_ptr(), c_uint32p)
        targets_ptr = cast(targets.data_ptr(), c_uint32p)
        rules_ptr = cast(rules.data_ptr(), c_uint32p)

        results = (c_float * len(rules))()
        results_ptr = cast(results, c_floatp)
        model(
            num_nodes,
            num_edges,
            num_rules,
            nodes_ptr,
            sources_ptr,
            targets_ptr,
            rules_ptr,
            results_ptr
        )
        policy = torch.softmax(torch.tensor(list(results)), dim=0)
        print(policy)
        print(y)
