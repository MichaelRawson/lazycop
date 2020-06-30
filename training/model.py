import torch
from torch.nn import BatchNorm1d, Embedding, Linear, Module, ModuleList, Parameter
from torch.nn.functional import relu
from torch.nn.init import xavier_normal_, zeros_

NODE_TYPES = 7
CHANNELS = 64
HIDDEN = 4096
MODULES = 4

class BN(BatchNorm1d):
    def __init__(self):
        super().__init__(CHANNELS, affine=False, track_running_stats=False)

class Conv(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.Tensor(CHANNELS, CHANNELS))
        self.reset_parameters()

    def reset_parameters(self):
        xavier_normal_(self.weight)

    def forward(self, x, sources, targets, norm):
        x = norm * x.index_add(0, targets, x[sources])
        return x @ self.weight

class BiConv(Module):
    def __init__(self):
        super().__init__()
        self.out = Conv()
        self.back = Conv()

    def forward(self, x, sources, targets, norm, norm_t):
        out = self.out(x, sources, targets, norm)
        back = self.back(x, targets, sources, norm_t)
        return out + back

class Residual(Module):
    def __init__(self):
        super().__init__()
        self.bn1 = BN()
        self.bn2 = BN()
        self.conv1 = BiConv()
        self.conv2 = BiConv()

    def forward(self, x, sources, targets, norm, norm_t):
        save = x
        x = relu(x)
        x = self.bn1(x)
        x = self.conv1(x, sources, targets, norm, norm_t)
        x = relu(x)
        x = self.bn2(x)
        x = self.conv2(x, sources, targets, norm, norm_t)
        return save + x

class Model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.conv = BiConv()
        self.res = ModuleList([Residual() for _ in range(MODULES)])
        self.hidden = Linear(CHANNELS, HIDDEN)
        self.fc = Linear(HIDDEN, 1)

    def forward(self, x, sources, targets, batch, counts, total):
        weights = torch.ones_like(sources, dtype=torch.float32)
        norm = 1.0 / torch.ones_like(x, dtype=torch.float32)\
            .scatter_add(0, targets, weights)\
            .unsqueeze(dim=1)
        norm_t = 1.0 / torch.ones_like(x, dtype=torch.float32)\
            .scatter_add(0, sources, weights)\
            .unsqueeze(dim=1)

        x = self.embedding(x)
        x = self.conv(x, sources, targets, norm, norm_t)
        for res in self.res:
            x = res(x, sources, targets, norm, norm_t)

        pooled = torch.zeros((total, CHANNELS), device=x.device)\
            .index_add_(0, batch, x)
        x = pooled / counts.unsqueeze(dim=1) 
        x = self.hidden(x)
        x = relu(x)
        x = self.fc(x)
        return x.squeeze()
