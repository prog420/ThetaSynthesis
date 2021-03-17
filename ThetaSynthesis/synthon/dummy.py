# -*- coding: utf-8 -*-
#
#  Copyright 2020-2021 Alexander Sizov <murkyrussian@gmail.com>
#  Copyright 2021 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of ThetaSynthesis.
#
#  ThetaSynthesis is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from .abc import SynthonABC
from CGRtools import smiles


class DummySynthon(SynthonABC):
    """
    Test synthon for Acetaminophen.
    """
    __slots__ = ()

    def __iter__(self):
        for prob, molecules in data[self._molecule][0]:
            yield prob, tuple(type(self)(mol) for mol in molecules)

    def __bool__(self):
        return self._molecule in building_blocks

    def __float__(self):
        return data[self._molecule][1]


building_blocks = {smiles('Oc1ccccc1')}
pre_data = {
    'CC(=O)Nc1ccc(O)cc1': (((.25, ('Oc1ccc(N)cc1',),),
                            (.15, ('C(Nc1ccc(OC)cc1)(C)=O',)),
                            (.2, ('ON=C(C)c1ccc(O)cc1',)),
                            (.25, ('Oc1ccc(O)cc1',)),
                            (.15, ('C(Nc1ccc(OC2OCCCC2)cc1)(C)=O',))), 1.),
    'Oc1ccc(N)cc1': (((.3, ('O=N(=O)c1ccc(O)cc1',)),
                      (.4, ('c1(N)ccc(OC)cc1',)),
                      (.3, ('c1(O)ccc(F)cc1',))), 1.),
    'O=N(=O)c1ccc(O)cc1': (((.35, ('Oc1ccccc1',)),
                            (.35, ('N(c1ccc(N)cc1)(=O)=O',)),
                            (.3, ('c1(N(=O)=O)cc(c(cc1)O)Br',))), 1.),
    'Oc1ccccc1': ((), 1.),
    'C(Nc1ccc(OC)cc1)(C)=O': ((), -1.),
    'Oc1ccc(O)cc1': ((), 0.),
    'C(Nc1ccc(OC2OCCCC2)cc1)(C)=O': ((), -1.),
    'c1(N)ccc(OC)cc1': ((), -1.),
    'c1(O)ccc(F)cc1': (((1., ('c1(OCOCCOC)ccc(F)cc1',)),), .1),
    'c1(OCOCCOC)ccc(F)cc1': ((), -1.),
    'c1(N(=O)=O)cc(c(cc1)O)Br': ((), -1.),
    'N(c1ccc(N)cc1)(=O)=O': (((1., ('C(c1c([N+](=O)[O-])ccc(c1)N)(=O)O',)),), -.1),
    'C(c1c([N+](=O)[O-])ccc(c1)N)(=O)O': ((), -1.),
    'ON=C(C)c1ccc(O)cc1': (((.5, ('O=C(C)c1ccc(O)cc1',)),
                            (.2, ('[Si](C(C)(C)C)(Oc1ccc(C(=NO)C)cc1)(C)C',)),
                            (.3, ('c1(c(ccc(C(=NO)C)c1)O)N',))), 1.),
    'c1(c(ccc(C(=NO)C)c1)O)N': (((1., ('c1(c(cc(C(=NO)C)cc1)N)OCOC',)),), -.2),
    'c1(c(cc(C(=NO)C)cc1)N)OCOC': ((), -1.),
    'O=C(C)c1ccc(O)cc1': (((.85, ('Oc1ccccc1',)),
                           (.15, ('[Si](C(C)(C)C)(Oc1ccc(C(=O)C)cc1)(C)C',))), 1.),
    '[Si](C(C)(C)C)(Oc1ccc(C(=O)C)cc1)(C)C': ((), -1.),
    '[Si](C(C)(C)C)(Oc1ccc(C(=NO)C)cc1)(C)C': ((), -1.),
}


def convert(dct):
    out = {}
    for k, (tuples, value) in dct.items():
        k = smiles(k)
        k.canonicalize()
        new_mols = []
        for prob, mols in tuples:
            synth = []
            for mol in mols:
                new_mol = smiles(mol)
                new_mol.canonicalize()
                synth.append(new_mol)
            new_mols.append((prob, tuple(synth)))
        out[k] = (tuple(new_mols), value)
    return out


data = convert(pre_data)


__all__ = ['DummySynthon']
