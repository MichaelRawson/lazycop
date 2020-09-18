from pathlib import Path
import sys

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model import Model
from data import Examples

MOMENTUM = 0.9
LR = 0.01
BATCH = 64
GAMMA = 0.99995
SAVE = 1e4
TEMPERATURE = -5

if __name__ == '__main__':
    model = Model().to('cuda')
    optimiser = Adam(model.parameters(), lr=LR)
    scheduler = ExponentialLR(optimiser, gamma=GAMMA)
    step = 0
    batch_loss = 0
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
            if step % SAVE == 0:
                for name, value in model.named_parameters():
                    writer.add_histogram(name.replace('.', '/'), value, step)
                torch.save(model.state_dict(), 'save.pt')

            nodes = nodes.to('cuda')
            sources = sources.to('cuda')
            targets = targets.to('cuda')
            batch = batch.to('cuda')
            min_score = scores.min()
            max_score = scores.max()
            if min_score == max_score:
                continue
            scores = (scores - min_score) / (max_score - min_score)
            y = torch.softmax(TEMPERATURE * scores, dim=0).to('cuda')

            try:
                raw = model(
                    nodes,
                    sources,
                    targets,
                    batch
                )
            except RuntimeError:
                continue

            log_softmax = torch.log_softmax(raw, dim=0)
            loss = -(y * log_softmax).sum() / BATCH
            loss.backward()
            batch_loss += loss.detach()
            step += 1

            if step % BATCH == 0:
                writer.add_scalar(
                    'training/loss',
                    batch_loss,
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
                batch_loss = 0
