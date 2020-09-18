import sys
import torch

from model import Model, LAYERS
from data import examples

if __name__ == '__main__':
    model = Model()
    model.load_state_dict(torch.load(sys.argv[1], map_location='cpu'))
    model.eval()

    for nodes, sources, targets, batch, scores in examples(sys.argv[2]):
        with torch.no_grad():
            logits = model(nodes, sources, targets, batch)
       
        print(logits)
        #print(torch.softmax(logits, dim=0))
        #print(scores)
