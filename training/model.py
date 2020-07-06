import torch
from torch.nn import BatchNorm1d, Embedding, Linear, Module, ModuleList, Parameter
from torch.nn.functional import relu
from torch.nn.init import xavier_normal_, zeros_

NODE_TYPES = 7
CHANNELS = 64
MODULES = 4

class BN(BatchNorm1d):
    def __init__(self, channels):
        super().__init__(channels, affine=False)

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
        self.bn1 = BN(CHANNELS)
        self.bn2 = BN(CHANNELS)
        self.conv1 = BiConv()
        self.conv2 = BiConv()

    def forward(self, x, sources, targets, norm, norm_t):
        save = x
        x = self.bn1(relu(x))
        x = self.conv1(x, sources, targets, norm, norm_t)
        x = self.bn2(relu(x))
        x = self.conv2(x, sources, targets, norm, norm_t)
        return save + x

def compute_norms(x, sources, targets):
        weights = torch.ones_like(sources, dtype=torch.float32)
        norm = 1.0 / torch.ones_like(x, dtype=torch.float32)\
            .scatter_add(0, targets, weights)\
            .unsqueeze(dim=1)
        norm_t = 1.0 / torch.ones_like(x, dtype=torch.float32)\
            .scatter_add(0, sources, weights)\
            .unsqueeze(dim=1)
        return norm, norm_t

class Model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.conv = BiConv()
        self.res = ModuleList([Residual() for _ in range(MODULES)])
        self.fc1 = Linear(CHANNELS, CHANNELS)
        self.fc2 = Linear(CHANNELS, CHANNELS)
        self.fc3 = Linear(CHANNELS, 1)

    def forward(self, x, sources, targets, batch, counts, total):
        norm, norm_t = compute_norms(x, sources, targets)
        x = self.embedding(x)
        x = self.conv(x, sources, targets, norm, norm_t)
        for res in self.res:
            x = res(x, sources, targets, norm, norm_t)
        pooled = torch.zeros((total, CHANNELS), device=x.device)\
            .index_add_(0, batch, x)
        x = pooled / counts.unsqueeze(dim=1) 
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()
