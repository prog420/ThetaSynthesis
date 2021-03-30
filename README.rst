ThetaSynthesis
--------------

Retrosynthesis analysis tool


API Example::

    from CGRtools import smiles, RDFWrite
    from ThetaSynthesis import RetroTree
    from ThetaSynthesis.synthon import RolloutSynthon


    target = smiles('CC(=O)NC1=CC=C(O)C=C1')
    target.canonicalize()

    tree = RetroTree(target, synthon_class=RolloutSynthon)

    with RDFWrite('acetaminophen.rdf') as f:
        for n in tree:
            path = tree.synthesis_path(n)
            for r in path:
                r.clean2d()
                f.write(r)

    print(tree.report())
