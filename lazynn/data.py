import json
import torch
import gzip

def examples(path, self_loops=True):
    with gzip.open(path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            # this fits on my GPU
            if len(record['nodes']) > 2e4:
                continue

            nodes = torch.tensor(record['nodes'])
            identity = list(range(len(nodes))) if self_loops else []
            sources = torch.tensor(identity + record['sources'])
            targets = torch.tensor(identity + record['targets'])
            scores = torch.tensor(record['scores'], dtype=torch.float32)
            batch = torch.tensor(record['batch'])
            yield nodes, sources, targets, batch, scores

class Examples(torch.utils.data.IterableDataset):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def __iter__(self):
        return examples(self.path)
