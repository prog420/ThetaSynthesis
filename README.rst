ThetaSynthesis
--------------

Retrosynthesis analysis tool


API Example::

    from CGRtools import smiles, RDFWrite
    from ThetaSynthesis import RetroTree
    from ThetaSynthesis.synthon import DummySynthon


    target = smiles('CC(=O)NC1=CC=C(O)C=C1')
    target.canonicalize()

    tree = RetroTree(target, synthon_class=DummySynthon)

    with RDFWrite('acetaminophen.rdf') as f:
        for path in tree:
            for r in path:
                f.write(r)

    print(tree.report())
