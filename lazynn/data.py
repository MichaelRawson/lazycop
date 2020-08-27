import json
import torch

def examples(path):
    with open(path, 'rb') as f:
        for line in f:
            record = json.loads(line)
            scores = torch.tensor(record['scores'], dtype=torch.float32)
            nodes = torch.tensor(record['nodes'])
            sources = torch.tensor(record['sources'])
            targets = torch.tensor(record['targets'])
            batch = torch.tensor(record['batch'])
            yield nodes, sources, targets, batch, scores

class Examples(torch.utils.data.IterableDataset):
    def __init__(self, path):
        self.path = path
        super().__init__()

    def __iter__(self):
        return examples(self.path)
