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
from CGRtools import MoleculeContainer, Reactor, smiles
from collections import deque
from io import TextIOWrapper
from pkg_resources import resource_stream
from typing import Set, FrozenSet
from ..abc import SynthonABC


class RolloutSynthon(SynthonABC):
    __slots__ = ('_depth', '_float')
    __bb__ = None
    __reactors__ = None

    def __new__(cls, molecule, *args, **kwargs):
        if cls.__bb__ is None:
            from .rules import rules
            bb = [smiles(x.strip()) for x in TextIOWrapper(resource_stream(__name__, 'data/building_blocks.smiles'))]
            for b in bb:  # recalculate canonic forms. prevent errors when CGRtools rules set changes.
                b.canonicalize()
            cls.__bb__ = frozenset(str(b) for b in bb)
            cls.__reactors__ = tuple((1., Reactor(x, delete_atoms=True, automorphism_filter=False, one_shot=False))
                                     for x in rules)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, molecule, /):
        super().__init__(molecule)
        self._float = None

    def __call__(self, finish=10, **kwargs):
        self._depth = finish

    def __float__(self):
        if self._float is not None:
            return self._float
        elif self:
            self._float = 1.
            return self._float
        molecule = self._molecule
        seen = set()
        max_depth = self._depth
        queue = deque([(molecule, 0)])
        while queue:
            curr, depth = queue.popleft()
            depth = depth + 1
            if depth > max_depth:
                self._float = -.5
                return self._float
            seen.add(curr)
            try:
                result = next(x for _, r in self._sorted() for x in r([curr])).products
            except StopIteration:
                self._float = -1.
                return self._float
            for mol in result:
                mol.kekule()
                mol.thiele()
            if seen.isdisjoint(result):
                queue.extend((x, depth) for x in result if str(x) not in self.__bb__)
        self._float = 1.
        return self._float

    def __iter__(self):
        if self:
            return
        molecule = self.molecule
        seen: Set[FrozenSet['MoleculeContainer']] = set()
        for prob, reactor in self._sorted():
            for reaction in reactor([molecule]):
                prods = [m for x in reaction.products for m in x.split()]
                for mol in prods:
                    mol.kekule()
                    mol.thiele()
                products = frozenset(mol for mol in prods)
                if products in seen:
                    continue
                seen.add(products)
                yield prob, tuple(type(self)(mol) for mol in products)

    def __bool__(self):
        return str(self._molecule) in self.__bb__

    def _sorted(self):
        return self.__reactors__


__all__ = ['RolloutSynthon']
