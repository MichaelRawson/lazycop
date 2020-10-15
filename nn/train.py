import sys

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import cross_entropy, log_softmax

from model import Model
from data import loader, upload

BATCH = 64
MAX_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
MAX_STEPS = int(6e5 / BATCH)

def compute_loss(model, example):
    nodes, sources, targets, rules, y = upload(example)
    logits = model(
        nodes,
        sources,
        targets,
        rules
    )
    return cross_entropy(
        logits.unsqueeze(dim=0),
        y.unsqueeze(dim=0)
    )

def validate(model):
    loss = 0
    total = 0
    for example in loader('../data/validate.gz'):
        with torch.no_grad():
            loss += compute_loss(model, example).detach()
        total += 1

    return loss / total

if __name__ == '__main__':
    torch.manual_seed(0)
    model = Model().to('cuda')
    optimiser = SGD(
        model.parameters(),
        lr=MAX_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    scheduler = OneCycleLR(
        optimiser,
        max_lr=MAX_LR,
        total_steps=MAX_STEPS,
        cycle_momentum=False
    )
    seen = 0
    summary = SummaryWriter()
    while True:
        summary.add_scalar(
            'loss/validation',
            validate(model),
            seen
        )

        batch = 0
        for example in loader('../data/train.gz'):
            if seen >= BATCH * MAX_STEPS:
                summary.add_scalar(
                    'loss/validation',
                    validate(model),
                    seen
                )
                torch.save(model.state_dict(), 'model.pt')
                print("done")
                import sys
                sys.exit(0)

            loss = compute_loss(model, example)
            loss = loss / BATCH
            loss.backward()
            batch += loss.detach()
            seen += 1

            if seen % BATCH == 0:
                summary.add_scalar(
                    'optim/LR',
                    optimiser.param_groups[0]['lr'],
                    seen
                )
                summary.add_scalar(
                    'loss/training',
                    batch,
                    seen
                )
                batch = 0
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()

        torch.save(model.state_dict(), 'model.pt')
