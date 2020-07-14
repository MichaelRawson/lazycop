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

BATCH_SIZE = 64
MOMENTUM = 0.9
LR = 0.01
GAMMA = 0.99999
SAVE_INTERVAL = 1e3

if __name__ == '__main__':
    model = Model().to('cuda')
    optimiser = SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        nesterov=True
    )
    scheduler = ExponentialLR(optimiser, gamma=GAMMA)
    step = 0
    writer = SummaryWriter()
    while True:
        examples = Examples(sys.argv[1], BATCH_SIZE)
        loader = DataLoader(
            examples,
            collate_fn=lambda x: x[0],
            num_workers=1,
            pin_memory=True
        )
        for example in loader:
            if step % SAVE_INTERVAL == 0:
                for name, value in model.named_parameters():
                    writer.add_histogram(name.replace('.', '/'), value, step)
                torch.save(model.state_dict(), 'model.pt')

            x, sources, targets, batch, counts, total, heuristic, actual =\
                example
            x = x.to('cuda')
            sources = sources.to('cuda')
            targets = targets.to('cuda')
            batch = batch.to('cuda')
            counts = counts.to('cuda')
            heuristic = heuristic.to('cuda')
            actual = actual.to('cuda')

            raw = model(x, sources, targets, batch, counts, total)
            predicted = heuristic + raw
            error = mse_loss(predicted, actual)
            error.backward()
            step += 1

            writer.add_scalar(
                'training/LR',
                optimiser.param_groups[0]['lr'],
                step
            )
            writer.add_scalar(
                'training/absolute error',
                error.sqrt(),
                step
            )
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step()
