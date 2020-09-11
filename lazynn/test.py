from ctypes import cast, c_uint, c_uint32, c_float, CDLL, POINTER
c_uint32p = POINTER(c_uint32)
c_floatp = POINTER(c_float)
import sys

import torch

from data import examples

so = CDLL('./libmodel.so')
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

def forward(sample):
    nodes, sources, targets, batch, scores = sample
    _num_graphs = int(batch.max()) + 1
    num_nodes = c_uint32(len(nodes))
    num_edges = c_uint32(len(sources))
    num_graphs = c_uint32(_num_graphs)
    nodes = nodes.int()
    sources = sources.int()
    targets = targets.int()
    batch = batch.int()
    nodes_ptr = cast(nodes.data_ptr(), c_uint32p)
    sources_ptr = cast(sources.data_ptr(), c_uint32p)
    targets_ptr = cast(targets.data_ptr(), c_uint32p)
    batch_ptr = cast(batch.data_ptr(), c_uint32p)

    results = (c_float * _num_graphs)()
    results_ptr = cast(results, c_floatp)
    model(
        num_nodes,
        num_edges,
        num_graphs,
        nodes_ptr,
        sources_ptr,
        targets_ptr,
        batch_ptr,
        results_ptr
    )
    return list(results)

if __name__ == '__main__':
    init()
    test = examples(sys.argv[1], self_loops=False)
    for sample in test:
        print(forward(sample))
