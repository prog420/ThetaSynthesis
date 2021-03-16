# -*- coding: utf-8 -*-
#
#  Copyright 2020-2021 Alexander Sizov <murkyrussian@gmail.com>
#  Copyright 2021 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of CGRtools.
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
from itertools import chain, zip_longest
from pkg_resources import resource_stream
from pickle import load

from .rules import RulesNet
from ..abc import SynthonABC


class RolloutSynthon(SynthonABC):
    __slots__ = ('_depth', )
    __net__ = None
    __bb__ = None

    def __new__(cls, molecule, *args, **kwargs):
        if cls.__net__ is None:
            cls.__net__ = RulesNet.load_from_checkpoint(resource_stream(__name__, 'data/net.ckpt'))
            cls.__net__.eval()
            cls.__bb__ = frozenset(load(resource_stream(__name__, 'data/bb.pickle')))
        return super().__new__(cls, *args, **kwargs)

    def __call__(self, finish=10, **kwargs):
        self._depth = finish

    def __float__(self):
        molecule = self._molecule
        for _ in range(self._depth):
            i = chain.from_iterable(zip_longest((tuple(reactor([molecule])) for _, reactor in self.__net__.get_reactors(molecule))))
            try:
                reaction = next(x for x in i if x)[0]
            except StopIteration:
                return -1.
            molecule = next(iter(reaction.products))
            if molecule in self.__bb__:
                return 1.
        return -1.

    def __iter__(self):
        molecule = self._molecule
        for prob, reactor in self.__net__.get_reactors(self._molecule):
            for reaction in reactor([molecule], automorphism_filter=False):
                mols = []
                for mol in reaction.products:
                    mol.kekule()
                    mol.thiele()
                    mols.append(mol)
                yield prob, tuple(type(self)(mol) for mol in mols)

    def __bool__(self):
        return self._molecule in self.__bb__


__all__ = ['RolloutSynthon']
