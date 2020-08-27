from pathlib import Path
import sys

import torch
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model import Model
from data import Examples

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR = 0.01
GAMMA = 0.99995
SAVE_INTERVAL = 1e3
TEMPERATURE = 10
BATCH = 32

if __name__ == '__main__':
    model = Model().to('cuda')
    optimiser = SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    scheduler = ExponentialLR(optimiser, gamma=GAMMA)
    step = 0
    total_loss = 0
    writer = SummaryWriter()
    while True:
        examples = Examples(sys.argv[1])
        loader = DataLoader(
            examples,
            collate_fn=lambda x: x[0],
            num_workers=1,
            pin_memory=True
        )
        for nodes, sources, targets, batch, scores in loader:
            if step % SAVE_INTERVAL == 0:
                for name, value in model.named_parameters():
                    writer.add_histogram(name.replace('.', '/'), value, step)
                torch.save(model.state_dict(), 'save.pt')

            nodes = nodes.to('cuda')
            sources = sources.to('cuda')
            targets = targets.to('cuda')
            batch = batch.to('cuda')
            scores = scores / scores.max()
            y = torch.softmax(TEMPERATURE * -scores, dim=0).to('cuda')

            raw = model(
                nodes,
                sources,
                targets,
                batch,
            )
            loss = -((y * torch.log_softmax(raw, dim=0)).sum()) / BATCH
            loss.backward()
            total_loss += loss.detach()
            step += 1

            if step % BATCH == 0:
                writer.add_scalar(
                    'training/loss',
                    total_loss,
                    step
                )
                writer.add_scalar(
                    'training/LR',
                    optimiser.param_groups[0]['lr'],
                    step
                )
                optimiser.step()
                optimiser.zero_grad()
                scheduler.step()
                total_loss = 0
