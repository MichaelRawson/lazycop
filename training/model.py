import torch
from torch.nn import BatchNorm1d, Embedding, Linear, Module, ModuleList, Parameter
from torch.nn.functional import leaky_relu as relu
from torch.nn.init import xavier_normal_

NODE_TYPES = 7
CHANNELS = 256
LAYERS = 4

class BN(BatchNorm1d):
    def __init__(self):
        super().__init__(CHANNELS, track_running_stats=False)

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
        return torch.max(out, back)

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
        self.conv = ModuleList([Residual() for _ in range(LAYERS)])
        self.fc = Linear(CHANNELS, 1)

    def forward(self, x, sources, targets, norm, norm_t):
        x = self.embedding(x)
        for conv in self.conv:
            x = conv(x, sources, targets, norm, norm_t)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x.squeeze()
