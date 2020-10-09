import sys
import torch
from tqdm import tqdm

from model import Model
from data import examples

if __name__ == '__main__':
    model = Model().to('cuda')
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()

    for nodes, sources, targets, rules, y in examples(sys.argv[2]):
        nodes = nodes.to('cuda')
        sources = sources.to('cuda')
        targets = targets.to('cuda')
        with torch.no_grad():
            logits = model(nodes, sources, targets, rules)

        policy = torch.softmax(logits, dim=0)
        print(policy)
        print(y)
