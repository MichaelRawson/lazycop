import gzip
import json
import torch

def norms(sources, targets):
    return (edge_index, edge_weight), (edge_index_t, edge_weight_t)

def examples(path):
    with gzip.open(path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            x = torch.tensor(record['nodes'])
            sources = torch.tensor(record['from'])
            targets = torch.tensor(record['to'])
            weights = torch.ones(sources.shape)
            degree = torch.ones(x.shape).scatter_add(0, sources, weights)
            degree_t = torch.ones(x.shape).scatter_add(0, targets, weights)
            norm = 1.0 / degree.unsqueeze(dim=1)
            norm_t = 1.0 / degree_t.unsqueeze(dim=1)
            y = (record['actual'] - record['heuristic']) / record['heuristic']
            y = torch.tensor(y)
            yield x, sources, targets, norm, norm_t, y
