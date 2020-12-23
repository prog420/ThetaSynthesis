from CGRtools import SDFRead, smiles
from ThetaSynthesis import CombineSynthon, RetroTree, StupidSynthon, SlowSynthon
from pickle import dump


def main():
    target = smiles('CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O')

    tree = RetroTree(target=target, class_name=SlowSynthon, stop_conditions={'depth_count': 10,
                                                                             'step_count': 2000, })

    a = list(tree)
    paths = [tuple(react for node in x if (react := node.reaction)) for n, x in enumerate(tree.dfs()) if n < 50]
    with open('paths.pickle', 'wb') as f:
        dump(paths, f)
    with open('solutions.pickle', 'wb') as f:
        dump(a, f)


if __name__ == '__main__':
    main()
