import torch
from torch.nn import Embedding, Linear, Module, ModuleList
from torch.nn.functional import relu

NODE_TYPES = 7
CHANNELS = 64
HIDDEN = 1024
LAYERS = 4

class Conv(Module):
    def __init__(self):
        super().__init__()
        self.weight = Linear(CHANNELS, CHANNELS, bias=False)
        torch.nn.init.xavier_normal_(self.weight.weight)

    def forward(self, x, sources, targets, norm):
        x = norm * x.index_add(0, targets, x[sources])
        return relu(self.weight(x))

class BiConv(Module):
    def __init__(self):
        super().__init__()
        self.out = Conv()
        self.back = Conv()

    def forward(self, x, sources, targets, norm, norm_t):
        out = self.out(x, sources, targets, norm)
        back = self.back(x, targets, sources, norm_t)
        return out + back

def compute_norms(nodes, sources, targets):
        weights = torch.ones_like(sources, dtype=torch.float32)
        norm = 1.0 / torch.ones_like(nodes, dtype=torch.float32)\
            .scatter_add(0, targets, weights)\
            .unsqueeze(dim=1)
        norm_t = 1.0 / torch.ones_like(nodes, dtype=torch.float32)\
            .scatter_add(0, sources, weights)\
            .unsqueeze(dim=1)
        return norm, norm_t

class Model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.conv = ModuleList([BiConv() for i in range(LAYERS)])
        self.hidden = Linear(CHANNELS, HIDDEN)
        self.output = Linear(HIDDEN, 1)

    def forward(
        self,
        nodes,
        node_counts,
        sources,
        targets,
        assignment,
        graph_count
    ):
        norm, norm_t = compute_norms(nodes, sources, targets)
        x = self.embedding(nodes)
        for conv in self.conv:
            x = x + conv(x, sources, targets, norm, norm_t)
        x = torch.zeros(
            (graph_count, CHANNELS),
            device=x.device
        ).index_add_(0, assignment, x) / node_counts.unsqueeze(dim=1)
        x = self.hidden(x)
        x = relu(x)
        x = self.output(x)
        return x.squeeze()
