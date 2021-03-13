from .abc import SynthonABC
from CGRtools import smiles

pre_data = {
    'CC(=O)Nc1ccc(O)cc1': [('Oc1ccc(N)cc1', 'C(Nc1ccc(OC)cc1)(C)=O', 'ON=C(C)c1ccc(O)cc1',
                            'Oc1ccc(O)cc1', 'C(Nc1ccc(OC2OCCCC2)cc1)(C)=O'), 1.],
    'Oc1ccc(N)cc1': [('O=N(=O)c1ccc(O)cc1', 'c1(N)ccc(OC)cc1', 'c1(O)ccc(F)cc1'), 1.],
    'O=N(=O)c1ccc(O)cc1': [('Oc1ccccc1', 'N(c1ccc(N)cc1)(=O)=O', 'c1(N(=O)=O)cc(c(cc1)O)Br'), 1.],
    'Oc1ccccc1': [(), 1.],
    'C(Nc1ccc(OC)cc1)(C)=O': [(), -1.],
    'Oc1ccc(O)cc1': [(), 0.],
    'C(Nc1ccc(OC2OCCCC2)cc1)(C)=O': [(), -1.],
    'c1(N)ccc(OC)cc1': [(), -1.],
    'c1(O)ccc(F)cc1': [('c1(OCOCCOC)ccc(F)cc1', ), .1],
    'c1(OCOCCOC)ccc(F)cc1': [(), -1.],
    'c1(N(=O)=O)cc(c(cc1)O)Br': [(), -1.],
    'N(c1ccc(N)cc1)(=O)=O': [('C(c1c([N+](=O)[O-])ccc(c1)N)(=O)O', ), -.1],
    'C(c1c([N+](=O)[O-])ccc(c1)N)(=O)O': [(), -1.],
    'ON=C(C)c1ccc(O)cc1': [('O=C(C)c1ccc(O)cc1', '[Si](C(C)(C)C)(Oc1ccc(C(=NO)C)cc1)(C)C',
                            'c1(c(ccc(C(=NO)C)c1)O)N'), 1.],
    'c1(c(ccc(C(=NO)C)c1)O)N': [('c1(c(cc(C(=NO)C)cc1)N)OCOC', ), -.2],
    'c1(c(cc(C(=NO)C)cc1)N)OCOC': [('c1(c([N+](=O)[O-])cc(C(=NO)C)cc1)OCOC', ), -.6],
    'c1(c([N+](=O)[O-])cc(C(=NO)C)cc1)OCOC': [(), -1.],
    'O=C(C)c1ccc(O)cc1': [('Oc1ccccc1', '[Si](C(C)(C)C)(Oc1ccc(C(=O)C)cc1)(C)C'), 1.],
    '[Si](C(C)(C)C)(Oc1ccc(C(=O)C)cc1)(C)C': [(), -1.],
}


class DummySynthon(SynthonABC):
    def __init__(self, molecule, /):
        super().__init__(molecule)

    def __iter__(self):
        for mol in data[self._molecule][0]:
            yield type(self)(mol)

    def __bool__(self):
        return self._molecule in building_blocks

    def __float__(self):
        return data[self._molecule][1]


building_blocks = {smiles('Oc1ccccc1')}


def convert(dct):
    out = {}
    for k, (mols, value) in dct.items():
        k = smiles(k)
        k.canonicalize()
        for mol in mols:
            mol = smiles(mol)
            mol.canonicalize()
        out[k] = [mols, value]
    return out


data = convert(pre_data)


__all__ = ['DummySynthon']
