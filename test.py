from CGRtools import SDFRead
from ThetaSynthesis import CombineSynthon, RetroTree, StupidSynthon, SlowSynthon


def main():
    with SDFRead('source files/sample.sdf', 'r') as file:
        targets = file.read()
    target = targets[19]
    tree = RetroTree(target=target, class_name=SlowSynthon, stop_conditions={'depth_count': 10,
                                                                             'step_count': 2000, })
    print(next(tree))


if __name__ == '__main__':
    main()
