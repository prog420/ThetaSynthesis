ThetaSynthesis
--------------

Resrosynthesis analysis tool


API Example::

    from CGRtools import smiles, RDFWrite
    from ThetaSynthesis import RetroTree
    from ThetaSynthesis.synthon import DummySynthon


    def main(target, output, synthon):
        target = smiles(target)
        target.canonicalize()

        tree = RetroTree(target, synthon_class=synthon)

        with RDFWrite(output) as f:
            for r in next(tree):
                f.write(r)


    if __name__ == '__main__':
        main('CC(=O)NC1=CC=C(O)C=C1', 'acetaminophen.rdf', DummySynthon)
