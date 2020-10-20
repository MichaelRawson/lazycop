import json
from glob import glob
import gzip
import random

def finish_proof(proofs, problem, proof):
    if problem not in proofs or len(proof) < len(proofs[problem]):
        proofs[problem] = proof

def do_file(proofs, f):
    problem = None
    proof = []
    for line in f:
        line = line.decode('ascii').strip()
        record = json.loads(line)
        if record['problem'] != problem:
            if problem != None:
                finish_proof(proofs, problem, proof)
            problem = record['problem']
            proof = []
        proof.append(line)
    finish_proof(proofs, problem, proof)

if __name__ == '__main__':
    proofs = {}
    for path in glob('gen*.gz'):
        with gzip.open(path) as f:
            do_file(proofs, f)

    data = [item for proof in proofs.values() for item in proof]
    random.shuffle(data)
    for item in data:
        print(item)
