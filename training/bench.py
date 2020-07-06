import torch
from tqdm import tqdm
from model import Model

if __name__ == '__main__':
    model = Model().to('cuda')
    model.eval()
    model.load_state_dict(torch.load('../model.pt'))
    nodes = torch.zeros(1000, dtype=torch.long).to('cuda')
    sources = (torch.arange(2000, dtype=torch.long) // 2).to('cuda')
    targets = (torch.arange(2000, dtype=torch.long) // 2).to('cuda')
    batch = (torch.arange(1000, dtype=torch.long) // 100).to('cuda')
    counts = torch.tensor([100] * 10).to('cuda')
    
    with torch.no_grad():
        for _ in tqdm(range(1000)):
            model(nodes, sources, targets, batch, counts, 10).cpu()
