from pathlib import Path
import sys

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from model import Model
from data import Examples

BATCH = 64
LR = 0.01
MOMENTUM = 0.9
DECAY = 1e-4
GAMMA = 0.99995

if __name__ == '__main__':
    torch.manual_seed(0)
    model = Model().to('cuda')
    optimiser = SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=DECAY,
        nesterov=True
    )
    scheduler = ReduceLROnPlateau(optimiser)
    step = 0
    total = 0
    batch_loss = 0
    total_loss = 0
    summary = SummaryWriter()
    while True:
        examples = Examples(sys.argv[1])
        loader = DataLoader(
            examples,
            collate_fn=lambda x: x[0],
            num_workers=1,
            pin_memory=True
        )
        summary.add_scalar(
            'LR',
            optimiser.param_groups[0]['lr'],
            step
        )
        for name, value in model.named_parameters():
            summary.add_histogram(name.replace('.', '/'), value, step)
        for nodes, sources, targets, rules, y in loader:
            nodes = nodes.to('cuda')
            sources = sources.to('cuda')
            targets = targets.to('cuda')
            rules = rules.to('cuda')
            y = y.to('cuda')
            logits = model(
                nodes,
                sources,
                targets,
                rules
            )
            loss = cross_entropy(
                logits.unsqueeze(dim=0),
                y.unsqueeze(dim=0)
            )
            total_loss += loss.detach()
            loss = loss / BATCH
            loss.backward()
            batch_loss += loss.detach()
            step += 1
            total += 1

            if step % BATCH == 0:
                summary.add_scalar(
                    'loss/batch',
                    batch_loss,
                    step
                )
                optimiser.step()
                optimiser.zero_grad()
                batch_loss = 0

        summary.add_scalar(
            'loss/total',
            total_loss / total,
            step
        )
        scheduler.step(total_loss)
        total = 0
        total_loss = 0
        torch.save(model.state_dict(), 'save.pt')
