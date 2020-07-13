import gzip
import json
import torch

class Batcher:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.x = []
        self.sources = []
        self.targets = []
        self.batch = []
        self.counts = []
        self.heuristic = []
        self.actual = []

    def append(self, x, sources, targets, heuristic, actual):
        self.total += 1
        self.batch.append(len(self.x) * torch.ones(x.shape, dtype=torch.long))
        self.x.append(x)
        self.sources.append(sources)
        self.targets.append(targets)
        self.counts.append(len(x))
        self.heuristic.append(heuristic)
        self.actual.append(actual)

    def finish_batch(self):
        x = torch.cat(self.x)
        sources = torch.cat(self.sources)
        targets = torch.cat(self.targets)
        batch = torch.cat(self.batch)
        counts = torch.tensor(self.counts).unsqueeze(dim=1)
        total = self.total
        heuristic = torch.tensor(self.heuristic)
        actual = torch.tensor(self.actual)
        self.reset()
        return (x, sources, targets, batch, counts, total, heuristic, actual)

def examples(path, batch_size):
    with gzip.open(path, 'rb') as f:
        batcher = Batcher()
        for line in f:
            record = json.loads(line)
            x = torch.tensor(record['nodes'])
            sources = torch.tensor(record['from'])
            targets = torch.tensor(record['to'])
            heuristic = float(record['heuristic'])
            actual = float(record['actual'])
            batcher.append(x, sources, targets, heuristic, actual)
            if batcher.total >= batch_size:
                yield batcher.finish_batch()

        if batcher.total > 0:
            yield batcher.finish_batch()

class Examples(torch.utils.data.IterableDataset):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        super().__init__()

    def __iter__(self):
        return examples(self.path, self.batch_size)
