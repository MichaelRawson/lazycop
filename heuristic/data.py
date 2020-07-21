import gzip
import json
import torch

class Batcher:
    def __init__(self):
        self.reset()

    def reset(self):
        self.node_count = 0
        self.edge_count = 0
        self.graph_count = 0
        self.assignment = []
        self.nodes = []
        self.node_counts = []
        self.sources = []
        self.targets = []
        self.heuristic = []
        self.estimate = []
        self.node_boundary = []
        self.edge_boundary = []

    def append(self, nodes, sources, targets, heuristic, estimate):
        self.assignment.append(
            self.graph_count * torch.ones(nodes.shape, dtype=torch.long)
        )
        self.nodes.append(nodes)
        self.node_counts.append(len(nodes))
        self.sources.append(sources + self.node_count)
        self.targets.append(targets + self.node_count)
        self.heuristic.append(heuristic)
        self.estimate.append(estimate)
        self.node_count += len(nodes)
        self.edge_count += len(sources)
        self.graph_count += 1
        self.node_boundary.append(self.node_count)
        self.edge_boundary.append(self.edge_count)

    def finish_batch(self):
        assignment = torch.cat(self.assignment)
        nodes = torch.cat(self.nodes)
        node_counts = torch.tensor(self.node_counts)
        sources = torch.cat(self.sources)
        targets = torch.cat(self.targets)
        heuristic = torch.tensor(self.heuristic)
        estimate = torch.tensor(self.estimate)
        node_boundary = torch.tensor(self.node_boundary)
        edge_boundary = torch.tensor(self.edge_boundary)
        graph_count = self.graph_count
        self.reset()
        return {
            'assignment': assignment,
            'nodes': nodes,
            'node_counts': node_counts,
            'sources': sources,
            'targets': targets,
            'heuristic': heuristic,
            'estimate': estimate,
            'node_boundary': node_boundary,
            'edge_boundary': edge_boundary,
            'graph_count': graph_count
        }

def examples(path, batch_size):
    with gzip.open(path, 'rb') as f:
        batcher = Batcher()
        for line in f:
            record = json.loads(line)
            x = torch.tensor(record['nodes'])
            sources = torch.tensor(record['from'])
            targets = torch.tensor(record['to'])
            heuristic = float(record['heuristic'])
            estimate = float(record['estimate'])
            batcher.append(x, sources, targets, heuristic, estimate)
            if batcher.graph_count >= batch_size:
                yield batcher.finish_batch()

        if batcher.graph_count > 0:
            yield batcher.finish_batch()

class Examples(torch.utils.data.IterableDataset):
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        super().__init__()

    def __iter__(self):
        return examples(self.path, self.batch_size)
