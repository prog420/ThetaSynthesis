from CGRtools import smiles
from ThetaSynthesis import RetroTree, SlowSynthon
from pickle import dump


def main():
    target = smiles('O=C(O)\C=C(\C=C\C=C(\C=C\c1c(cc(OC)c(c1C)C)C)C)C')
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
