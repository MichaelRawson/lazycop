import sys

import torch
from tqdm import tqdm
from model import Model

if __name__ == '__main__':
    model = Model().to('cuda')
    model.eval()
    model.load_state_dict(torch.load(sys.argv[1]))
    nodes = torch.zeros(1000, dtype=torch.long)
    sources = (torch.arange(2000, dtype=torch.long) // 2)
    targets = (torch.arange(2000, dtype=torch.long) // 2)
    batch = (torch.arange(1000, dtype=torch.long) // 100)
    counts = torch.tensor([100.0] * 10).unsqueeze(dim=1)
    
    with torch.no_grad():
        for _ in tqdm(range(1000)):
            model(
                    nodes.to('cuda'),
                    sources.to('cuda'),
                    targets.to('cuda'),
                    batch.to('cuda'),
                    counts.to('cuda'),
                    10
            ).cpu()
            torch.cuda.current_stream().synchronize()
