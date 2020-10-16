import sys

import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cross_entropy, log_softmax
from torch_scatter import scatter_log_softmax

from model import Model
from data import loader, upload

BATCH = 64
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

def compute_loss(model, example):
    nodes, sources, targets, rules, graph, y = upload(example)
    logits = model(
        nodes,
        sources,
        targets,
        rules
    )
    log_softmax = scatter_log_softmax(logits, graph, dim=0)
    return -log_softmax[y].mean()

def validate(model):
    loss = 0
    count = 0
    for example in loader('../data/validate.gz', BATCH):
        with torch.no_grad():
            loss += compute_loss(model, example)
        count += 1
    return loss / count

if __name__ == '__main__':
    torch.manual_seed(0)
    model = Model().to('cuda')
    optimiser = SGD(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        momentum=MOMENTUM,
        nesterov=True
    )
    seen = 0
    best = float('inf')
    summary = SummaryWriter()
    while True:
        for example in loader('../data/train.gz', BATCH):
            loss = compute_loss(model, example)
            loss.backward()
            seen += 1
            optimiser.step()
            optimiser.zero_grad()
            summary.add_scalar(
                'loss/training',
                loss,
                seen
            )

        validation = validate(model)
        summary.add_scalar(
            'loss/validation',
            validation,
            seen
        )
        if validation < best:
            best = validation
            torch.save(model.state_dict(), 'model.pt')
