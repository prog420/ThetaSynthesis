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
    elif line.startswith('#'):
        continue
    elif target is None:
        target = smiles(line)
        target.canonicalize()
    else:
        r = smiles(line)
        r.canonicalize()
        reactions.add(r)

for num in [.01, .1, 1., 10., 100.]:
    results = []
    for target, reactions in data:
        found = []
        tree = RetroTree(target, synthon_class=RolloutSynthon, size=10000, c_puct=num)
        for node in tree:
            path = tree.synthesis_path(node)
            if reactions.issuperset(path):
                found.append((True, path))
            else:
                found.append((False, path))
        results.append((target, found))

    with open(f'results_puct_{num}.pkl', 'wb') as f:
        dump(results, f)
