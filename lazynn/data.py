import json
import torch
import gzip

def examples(path, self_loops=True):
    with gzip.open(path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            nodes = torch.tensor(record['nodes'])
            identity = list(range(len(nodes))) if self_loops else []
            sources = torch.tensor(identity + record['sources'])
            targets = torch.tensor(identity + record['targets'])
            rules = torch.tensor(record['rules'])
            y = torch.tensor(record['y'])
            yield nodes, sources, targets, rules, y

class Examples(torch.utils.data.IterableDataset):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def __iter__(self):
        return examples(self.path)
