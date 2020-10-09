import sys

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cross_entropy, log_softmax

from model import Model
from data import loader, upload

BATCH = 64
ENTROPY_REGULARISATION = 0.5

def compute_loss(model, example):
    nodes, sources, targets, rules, y = upload(example)
    logits = model(
        nodes,
        sources,
        targets,
        rules
    )
    xentropy = cross_entropy(
        logits.unsqueeze(dim=0),
        y.unsqueeze(dim=0)
    )
    entropy = (
        torch.softmax(logits, dim=0) *
        log_softmax(logits, dim=0)
    ).sum()
    return xentropy + ENTROPY_REGULARISATION * entropy

def validate(model):
    model.eval()
    total = 0
    for example in loader('../data/validate.gz'):
        with torch.no_grad():
            total += compute_loss(model, example)
    model.train()
    return total

if __name__ == '__main__':
    torch.manual_seed(0)
    model = Model().to('cuda')
    optimiser = AdamW(model.parameters())
    summary = SummaryWriter()
    step = 0
    best = validate(model)
    while True:
        summary.add_scalar(
            'validation',
            best,
            step
        )
        for example in loader('../data/train.gz'):
            loss = compute_loss(model, example)
            summary.add_scalar(
                'loss',
                loss.detach(),
                step
            )
            loss = loss / BATCH
            loss.backward()
            step += 1
            if step % BATCH == 0:
                optimiser.step()
                optimiser.zero_grad()

        new = validate(model)
        if new >= best:
            print("done")
            break

        best = new
        torch.save(model.state_dict(), 'model.pt')
