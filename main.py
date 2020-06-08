from CGRtools.files import SDFRead
from ThetaSynthesis import RetroTree, CombineSynthon


def main():
    with SDFRead('25.sdf', 'r') as file:
        targets = file.read()
    target = targets[4]
    tree = RetroTree(CombineSynthon(target), {'step_count': 10000, 'depth_count': 10, 'terminal_count': 1000})
    paths = list(tree)
    print(paths)


if __name__ == '__main__':
    main()
