from pathlib import Path
import sys

import torch
from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model import Model
from data import Examples

BATCH_SIZE = 64
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR = 0.01
GAMMA = 0.999995
SAVE_INTERVAL = 1e3
CLIP_NORM = 10

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
    writer = SummaryWriter()
    while True:
        examples = Examples('data.gz', BATCH_SIZE)
        loader = DataLoader(
            examples,
            collate_fn=lambda x: x[0],
            num_workers=1,
            pin_memory=True
        )
        for batch in loader:
            if step % SAVE_INTERVAL == 0:
                for name, value in model.named_parameters():
                    writer.add_histogram(name.replace('.', '/'), value, step)
                torch.save(model.state_dict(), 'save.pt')

            nodes = batch['nodes'].to('cuda')
            node_counts = batch['node_counts'].to('cuda')
            sources = batch['sources'].to('cuda')
            targets = batch['targets'].to('cuda')
            assignment = batch['assignment'].to('cuda')
            heuristic = batch['heuristic'].to('cuda')
            estimate = batch['estimate'].to('cuda')
            graph_count = batch['graph_count']

            raw = model(
                nodes,
                node_counts,
                sources,
                targets,
                assignment,
                graph_count
            )
            predicted = heuristic + raw
            loss = mse_loss(predicted, estimate)
            loss.backward()
            clip_grad_norm_(model.parameters(), CLIP_NORM)
            step += 1

            writer.add_scalar(
                'training/LR',
                optimiser.param_groups[0]['lr'],
                step
            )
            writer.add_scalar(
                'training/loss',
                loss.detach(),
                step
            )
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step()
