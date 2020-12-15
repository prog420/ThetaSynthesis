from CGRtools import SDFRead
from pickle import dump
from ThetaSynthesis import CombineSynthon, RetroTree, StupidSynthon, SlowSynthon
import matplotlib


def main():
    with SDFRead('source files/sample.sdf', 'r') as file:
        targets = file.read()
    target = targets[9]
    tree = RetroTree(target=target, class_name=SlowSynthon, stop_conditions={'depth_count': 10,
                                                                             'step_count': 200, })
    print(next(tree))
    print(next(tree))


if __name__ == '__main__':
    main()

