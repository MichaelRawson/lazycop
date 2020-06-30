from pathlib import Path
import sys
from data import examples
from model import Model
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss

SAVE_INTERVAL = 10000
BATCH_SIZE = 16
BASE_LR = 0
MAX_LR = 5e-3
GAMMA = 0.99999
HALF_CYCLE = 5000
WEIGHT_DECAY = 1e-4

if __name__ == '__main__':
    model = Model().to('cuda')
    optimiser = SGD(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        momentum=0.9,
        nesterov=True
    )
    scheduler = CyclicLR(
        optimiser,
        BASE_LR,
        MAX_LR,
        mode='exp_range',
        gamma=GAMMA,
        step_size_up=HALF_CYCLE
    )
    step = 0
    writer = SummaryWriter()
    while True:
        for example in examples(sys.argv[1], BATCH_SIZE):
            x, sources, targets, batch, counts, total, y = example
            x = x.to('cuda')
            sources = sources.to('cuda')
            targets = targets.to('cuda')
            batch = batch.to('cuda')
            counts = counts.to('cuda')
            y = y.to('cuda')
            params = (x, sources, targets, batch, counts, total)

            predicted = model(*params)
            error = mse_loss(predicted, y)
            error.backward()
            step += 1

            writer.add_scalar(
                'absolute error',
                error.sqrt(),
                step
            )
            writer.add_scalar(
                'LR',
                scheduler.get_last_lr()[0],
                step
            )
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step()

            if step % SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), 'model.pt')
