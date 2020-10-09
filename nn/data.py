import json
import gzip

import torch
from torch.utils.data import DataLoader

def upload(example):
    nodes, sources, targets, rules, y = example
    nodes = nodes.to('cuda')
    sources = sources.to('cuda')
    targets = targets.to('cuda')
    rules = rules.to('cuda')
    y = y.to('cuda')
    return nodes, sources, targets, rules, y

def examples(path, self_loops=True, validate=False):
    with gzip.open(path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            nodes = torch.tensor(record['nodes'])
            sources = torch.tensor(record['sources'])
            targets = torch.tensor(record['targets'])
            rules = torch.tensor(record['rules'])
            y = torch.tensor(record['y'])
            if self_loops:
                identity = torch.arange(len(nodes))
                sources = torch.cat((identity, sources))
                targets = torch.cat((identity, targets))
            yield nodes, sources, targets, rules, y

class Examples(torch.utils.data.IterableDataset):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def __iter__(self):
        return examples(self.path)

def loader(path):
    examples = Examples(path)
    return DataLoader(
        examples,
        collate_fn=lambda x: x[0],
        num_workers=1,
        pin_memory=True
    )
