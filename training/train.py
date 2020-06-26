from pathlib import Path
import sys
from data import examples
from model import Model
from torch import jit
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 64
SAVE_INTERVAL = 10000
BASE_LR = 1e-6
MAX_LR = 1e-4
HALF_CYCLE = 5000 / BATCH_SIZE
WEIGHT_DECAY = 1e-4

if __name__ == '__main__':
    model = jit.script(Model().to('cuda'))
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
        step_size_up=HALF_CYCLE
    )
    step = 0
    writer = SummaryWriter()
    while True:
        for x, sources, targets, norm, norm_t, y in examples(sys.argv[1]):
            x = x.to('cuda')
            sources = sources.to('cuda')
            targets = targets.to('cuda')
            norm = norm.to('cuda')
            norm_t = norm_t.to('cuda')
            y = y.to('cuda')
            params = (
                x,
                sources,
                targets,
                norm,
                norm_t
            )

            predicted = model(*params)
            error = ((y - predicted) ** 2) / BATCH_SIZE
            error.backward()
            step += 1

            writer.add_scalar(
                'absolute error',
                (y - predicted).abs(),
                step
            )
            writer.add_scalar(
                'LR',
                scheduler.get_last_lr()[0],
                step
            )

            if step % BATCH_SIZE == 0:
                optimiser.step()
                optimiser.zero_grad()
                scheduler.step()

            if step % SAVE_INTERVAL == 0:
                model.save('model.pt')
