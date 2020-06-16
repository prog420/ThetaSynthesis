from CGRtools import SDFRead
from cProfile import Profile, run
from pstats import Stats
from pickle import dump
from ThetaSynthesis import CombineSynthon, RetroTree, StupidSynthon, SlowSynthon


def main():
    with SDFRead('source files/sample.sdf', 'r') as file:
        targets = file.read()
    target = targets[6]
    tree = RetroTree(target=target, class_name=CombineSynthon, stop_conditions={'depth_count': 10,
                                                                                'step_count': 200,
                                                                                })
    print(next(tree))


if __name__ == '__main__':
    main()
