import json
import gzip

import torch
from torch.utils.data import DataLoader

def upload(example):
    return tuple(item.to('cuda') for item in example)

def examples(path, self_loops=True, validate=False):
    with gzip.open(path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            nodes = torch.tensor(record['nodes'])
            sources = torch.tensor(record['sources'])
            targets = torch.tensor(record['targets'])
            rules = torch.tensor(record['rules'])
            y = torch.tensor([record['y']])
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

def collate(batch):
    bnodes = []
    bsources = []
    btargets = []
    brules = []
    bgraph = []
    by = []
    count = 0
    node_offset = 0
    rule_offset = 0
    for nodes, sources, targets, rules, y in batch:
        items = len(nodes)
        bnodes.append(nodes)
        bsources.append(sources + node_offset)
        btargets.append(targets + node_offset)
        brules.append(rules + node_offset)
        bgraph.append(torch.full((len(rules),), count, dtype=torch.long))
        by.append(y + rule_offset)
        node_offset += len(nodes)
        rule_offset += len(rules)
        count += 1

    nodes = torch.cat(bnodes)
    sources = torch.cat(bsources)
    targets = torch.cat(btargets)
    rules = torch.cat(brules)
    graph = torch.cat(bgraph)
    y = torch.cat(by)
    batch = (nodes, sources, targets, rules, graph, y)
    return batch

def loader(path, batch):
    examples = Examples(path)
    return DataLoader(
        examples,
        collate_fn=collate,
        batch_size=batch,
        num_workers=1,
        pin_memory=True
    )
