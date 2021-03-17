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
from CGRtools import smiles, Reactor, ReactionContainer


class GreedySynthon(SynthonABC):
    """
    Test synthon for Acetaminophen.
    """
    __slots__ = ()

    def __iter__(self):
        for prob, reactor, synth in data[self._molecule][0]:
            mols = []
            for r in reactor([self.molecule], automorphism_filter=False):
                if synth != set(r.products):
                    continue
                for mol in r.products:
                    # fix hydrogens
                    mol.kekule()
                    mol.thiele()
                    mols.append(mol)
                yield prob, tuple(type(self)(mol) for mol in mols)
                break

    def __bool__(self):
        return self._molecule in building_blocks

    def __float__(self):
        return data[self._molecule][1]


building_blocks = {smiles('Oc1ccccc1')}
pre_data = {
    '[CH3:1][C:2](=[O:3])[NH:4][c:5]1[cH:6][cH:7][c:8]([OH:9])[cH:10][cH:11]1': (
            ((.25, ('[NH2:4][c:5]1[cH:6][cH:7][c:8]([OH:9])[cH:10][cH:11]1',),),
             (.15, ('[CH3:19][O:9][c:8]1[cH:7][cH:6][c:5]([NH:4][C:2]([CH3:1])=[O:3])[cH:11][cH:10]1',)),
             (.2, ('[CH3:1][C:2](=[N:4][OH:3])[c:5]1[cH:6][cH:7][c:8]([OH:9])[cH:10][cH:11]1',)),
             (.25, ('[OH:9][c:8]1[cH:10][cH:11][c:5]([OH:12])[cH:6][cH:7]1',)),
             (.15, ('[CH3:1][C:2](=[O:3])[NH:4][c:5]1[cH:6][cH:7][c:8]([O:9][CH:21]2[CH2:16][CH2:17][CH2:18][CH2:19][O:20]2)[cH:10][cH:11]1',))),
            1.),
    '[NH2:4][c:5]1[cH:6][cH:7][c:8]([OH:9])[cH:10][cH:11]1': (
            ((.3, ('[OH:9][c:8]1[cH:7][cH:6][c:5]([cH:11][cH:10]1)[N:4](=[O:3])=[O:1]',)),
             (.4, ('[CH3:15][O:9][c:8]1[cH:7][cH:6][c:5]([NH2:4])[cH:11][cH:10]1',)),
             (.3, ('[OH:9][c:8]1[cH:7][cH:6][c:5]([F:14])[cH:11][cH:10]1',))), 1.),
    '[OH:9][c:8]1[cH:7][cH:6][c:5]([cH:11][cH:10]1)[N+:4]([O-:3])=[O:1]': (
            ((.35, ('[OH:9][c:8]1[cH:10][cH:11][cH:5][cH:6][cH:7]1',)),
             (.35, ('N[c:8]1[cH:7][cH:6][c:5]([cH:11][cH:10]1)[N:4](=[O:1])=[O:3]',)),
             (.3, ('[OH:9][c:8]1[cH:7][cH:6][c:5]([cH:11][c:10]1Br)[N+:4](=[O:1])[O-:3]',))), 1.),
    'Oc1ccccc1': ((), 1.),
    'C(Nc1ccc(OC)cc1)(C)=O': ((), -1.),
    'Oc1ccc(O)cc1': ((), 0.),
    'C(Nc1ccc(OC2OCCCC2)cc1)(C)=O': ((), -1.),
    'c1(N)ccc(OC)cc1': ((), -1.),
    '[OH:9][c:8]1[cH:7][cH:6][c:5]([F:14])[cH:11][cH:10]1': (((1., ('COCCOC[O:9][c:8]1[cH:7][cH:6][c:5]([F:14])[cH:11][cH:10]1',)),), .1),
    'c1(OCOCCOC)ccc(F)cc1': ((), -1.),
    'c1(N(=O)=O)cc(c(cc1)O)Br': ((), -1.),
    '[NH2:30][c:8]1[cH:7][cH:6][c:5]([cH:11][cH:10]1)[N+:4](=[O:1])[O-:3]': (((1., ('[NH2:30][c:8]1[cH:10][cH:11][c:5]([c:6]([cH:7]1)C(O)=O)[N+:4]([O-:3])=[O:1]',)),), -.1),
    'C(c1c([N+](=O)[O-])ccc(c1)N)(=O)O': ((), -1.),
    '[CH3:1][C:2](=[N:4][OH:3])[c:5]1[cH:6][cH:7][c:8]([OH:9])[cH:10][cH:11]1': (
            ((.5, ('[CH3:1][C:2](=[O:3])[c:5]1[cH:6][cH:7][c:8]([OH:9])[cH:10][cH:11]1',)),
             (.2, ('[CH3:1][C:2](=[N:4][OH:3])[c:5]1[cH:11][cH:10][c:8]([O:9][Si](C)(C)C(C)(C)C)[cH:7][cH:6]1',)),
             (.3, ('[CH3:1][C:2](=[N:4][OH:3])[c:5]1[cH:11][cH:10][c:8]([OH:9])[c:7](N)[cH:6]1',))), 1.),
    '[CH3:1][C:2](=[N:4][OH:3])[c:5]1[cH:11][cH:10][c:8]([OH:9])[c:7](N)[cH:6]1': (((1., ('COC[O:9][c:8]1[cH:10][cH:11][c:5]([cH:6][c:7]1[NH2:23])[C:2]([CH3:1])=[N:4][OH:3]',)),), -.2),
    'c1(c(cc(C(=NO)C)cc1)N)OCOC': ((), -1.),
    '[CH3:1][C:2](=[O:3])[c:5]1[cH:6][cH:7][c:8]([OH:9])[cH:10][cH:11]1': (
            ((.85, ('[OH:9][c:8]1[cH:10][cH:11][cH:5][cH:6][cH:7]1',)),
             (.15, ('[CH3:1][C:2](=[O:3])[c:5]1[cH:11][cH:10][c:8]([O:9][Si](C)(C)C(C)(C)C)[cH:7][cH:6]1',))), 1.),
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
            rxn = ReactionContainer((k,), synth)
            ext_center = set(rxn.extended_centers_list[0])
            qk = k.substructure(ext_center.intersection(k), as_query=True)
            qsynth = [m.substructure(ext_center.intersection(m), as_query=True) for m in synth]
            template = ReactionContainer((qk,), qsynth)
            new_mols.append((prob, Reactor(template), set(synth)))
        out[k] = (tuple(new_mols), value)
    return out


data = convert(pre_data)


__all__ = ['GreedySynthon']
