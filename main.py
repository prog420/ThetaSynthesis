from CGRtools import smiles
from ThetaSynthesis import RetroTree, SlowSynthon
from pickle import dump


def main():
    target = smiles('CC(NC)CC1=CC=C(OCO2)C2=C1')
    target.canonicalize()

    tree = RetroTree(target=target, class_name=SlowSynthon, stop_conditions={'depth_count': 10,
                                                                             'step_count': 10000, })

    a = list(tree)
    with open('test.pickle', 'wb') as f:
        dump(a, f)

    c = list(tree.generate_examples())
    with open('examples.pickle', 'wb') as f:
        dump(c, f)


if __name__ == '__main__':
    main()
