from CGRtools import SDFRead, smiles
from ThetaSynthesis import CombineSynthon, RetroTree, StupidSynthon, SlowSynthon
from time import time


def main():
    with SDFRead('source files/sample.sdf', 'r') as file:
        targets = file.read()
    target = smiles('CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O')

    tree = RetroTree(target=target, class_name=StupidSynthon, stop_conditions={'depth_count': 10,
                                                                             'step_count': 10000, })

    now = time()
    a = list(tree)
    print(f'{time() - now}')
    print(a)


if __name__ == '__main__':
    main()
