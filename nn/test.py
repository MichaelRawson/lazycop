import sys
import torch
from tqdm import tqdm

from model import Model
from data import examples, upload

if __name__ == '__main__':
    model = Model().to('cuda')
    model.load_state_dict(torch.load(sys.argv[1]))
    model.eval()

    for example in examples(sys.argv[2]):
        nodes, sources, targets, rules, y = upload(example)
        with torch.no_grad():
            logits = model(nodes, sources, targets, rules)

        policy = torch.softmax(logits, dim=0)
        print(len(policy), policy[y])
