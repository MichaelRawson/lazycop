import sys

import torch
from tqdm import tqdm

from model import Model
from data import examples

if __name__ == '__main__':
    model = Model().to('cuda')
    model.eval()
    model.load_state_dict(torch.load('model.pt'))

    loader = examples('data.gz', 64)
    batch = next(loader)
   
    with torch.no_grad():
        for _ in tqdm(range(1000)):
            nodes = batch['nodes'].to('cuda')
            node_counts = batch['node_counts'].to('cuda')
            sources = batch['sources'].to('cuda')
            targets = batch['targets'].to('cuda')
            assignment = batch['assignment'].to('cuda')
            graph_count = batch['graph_count']
            print(model(
                nodes,
                node_counts,
                sources,
                targets,
                assignment,
                graph_count
            ).cpu())
