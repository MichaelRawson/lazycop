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
        self.y = []

    def append(self, x, sources, targets, y):
        self.total += 1
        self.batch.append(len(self.x) * torch.ones(x.shape, dtype=torch.long))
        self.x.append(x)
        self.sources.append(sources)
        self.targets.append(targets)
        self.counts.append(len(x))
        self.y.append(y)

    def finish_batch(self):
        x = torch.cat(self.x)
        sources = torch.cat(self.sources)
        targets = torch.cat(self.targets)
        batch = torch.cat(self.batch)
        counts = torch.tensor(self.counts)
        total = self.total
        y = torch.tensor(self.y)
        self.reset()
        return (x, sources, targets, batch, counts, total, y)

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
            y = actual - heuristic
            batcher.append(x, sources, targets, y)
            if batcher.total > batch_size:
                yield batcher.finish_batch()

        yield batcher.finish_batch()
