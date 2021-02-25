from CGRtools import SDFRead, smiles
from ThetaSynthesis import CombineSynthon, RetroTree, StupidSynthon, SlowSynthon
from pickle import dump


def main():
    target = smiles('CCCC(C(=O)C1=CC=CC=C1)N2CCCC2')
    target.canonicalize()

    tree = RetroTree(target=target, class_name=SlowSynthon, stop_conditions={'depth_count': 10,
                                                                             'step_count': 100, })

    a = list(tree)
    print(a)
    visits = [x.visit_count for x in tree.successors(tree._target)]
    print(visits)


if __name__ == '__main__':
    main()
