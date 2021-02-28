from CGRtools import smiles
from ThetaSynthesis import RetroTree, SlowSynthon


def main():
    target = smiles('C(c1ccc(CNC(c2c(OC)ccc(CC(C(=O)O)OCC)c2)=O)cc1)(F)(F)F')
    target.canonicalize()

    tree = RetroTree(target=target, class_name=SlowSynthon, stop_conditions={'depth_count': 10,
                                                                             'step_count': 10000, })

    a = list(tree)


if __name__ == '__main__':
    main()
