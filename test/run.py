from CGRtools import smiles
from ThetaSynthesis import RetroTree
from ThetaSynthesis.synthon import RolloutSynthon
from pickle import dump


data = []
target = None
reactions = set()
for line in open('test.smiles', 'r'):
    line = line.strip()
    if line == '$$$$':
        data.append((target, reactions))
        target = None
        reactions = set()
    elif target is None:
        target = smiles(line)
        target.canonicalize()
    else:
        r = smiles(line)
        r.canonicalize()
        reactions.add(r)


results = []
for target, reactions in data:
    found = []
    tree = RetroTree(target, synthon_class=RolloutSynthon, size=4000)
    for node in tree:
        path = tree.synthesis_path(node)
        if reactions.issuperset(path):
            found.append((True, path))
        else:
            found.append((False, path))
    results.append((target, found))

with open('results.pkl', 'wb') as f:
    dump(results, f)
