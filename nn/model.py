import torch
from torch.nn import BatchNorm1d, Embedding, Linear, Module, ModuleList, Parameter
from torch.nn.functional import relu_
from torch.nn.init import xavier_normal_
from torch_scatter import scatter_mean

NODE_TYPES = 19
CHANNELS = 64
HIDDEN = 1024
LAYERS = 24

class Conv(Module):
    def __init__(self):
        super().__init__()
        self.bn = BatchNorm1d(CHANNELS)
        self.weight = Linear(CHANNELS, CHANNELS)
        xavier_normal_(self.weight.weight)

    def forward(self, x, sources, targets):
        x = scatter_mean(x[sources], targets, dim_size=x.shape[0], dim=0)
        if self.bn is not None:
            x = self.bn(x)
        return relu_(self.weight(x))

    def fuse(self):
        denominator = (self.bn.running_var + self.bn.eps).rsqrt()
        self.weight.bias.data += self.weight.weight @ self.bn.bias
        self.weight.weight.data *= self.bn.weight * denominator
        self.weight.bias.data -= self.weight.weight.data @ self.bn.running_mean
        self.bn = None

class BiConv(Module):
    def __init__(self):
        super().__init__()
        self.out = Conv()
        self.back = Conv()

    def forward(self, x, sources, targets):
        out = self.out(x, sources, targets)
        back = self.back(x, targets, sources)
        return out + back

    def fuse(self):
        self.out.fuse()
        self.back.fuse()

class Model(Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(NODE_TYPES, CHANNELS)
        self.conv = ModuleList([BiConv() for i in range(LAYERS)])
        self.hidden = Linear(CHANNELS, HIDDEN)
        self.output = Linear(HIDDEN, 1, bias=False)

    def fuse(self):
        for conv in self.conv:
            conv.fuse()

    def forward(
        self,
        nodes,
        sources,
        targets,
        rules
    ):
        x = self.embedding(nodes)
        for conv in self.conv:
            x = x + conv(x, sources, targets)
        x = x[rules]
        x = relu_(self.hidden(x))
        return self.output(x).squeeze()
