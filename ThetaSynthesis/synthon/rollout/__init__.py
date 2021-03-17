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
from collections import deque
from pkg_resources import resource_stream
from pickle import load
from typing import Tuple, TYPE_CHECKING
from .rules import RulesNet
from ..abc import SynthonABC

if TYPE_CHECKING:
    from CGRtools import MoleculeContainer


class RolloutSynthon(SynthonABC):
    __slots__ = ('_depth', '_float', '__dict__')
    __net__ = None
    __bb__ = None

    def __new__(cls, molecule, *args, **kwargs):
        if cls.__net__ is None:
            cls.__net__ = RulesNet.load_from_checkpoint(resource_stream(__name__, 'data/net.ckpt'))
            cls.__net__.eval()
            cls.__bb__ = frozenset(load(resource_stream(__name__, 'data/bb.pickle')))
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, molecule, /):
        super().__init__(molecule)
        self._float = None

    def __call__(self, finish=10, **kwargs):
        self._depth = finish

    def __float__(self):
        if self._float is not None:
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
            result = self._get_products(curr)
            if not result:
                self._float = -1.
                return self._float
            queue.extend((x, depth) for x in set(result).difference(seen) if str(x) not in self.__bb__)
        self._float = 1.
        return self._float

    def _get_products(self, molecule: 'MoleculeContainer') -> Tuple['MoleculeContainer', ...]:
        for _, reactor in self.__net__.get_reactors(molecule):
            for r in reactor([molecule]):
                return r.products
        return ()

    def __iter__(self):
        molecule = self._molecule
        for prob, reactor in self.__net__.get_reactors(self._molecule):
            for reaction in reactor([molecule], automorphism_filter=False):
                for mol in reaction.products:
                    mol.kekule()
                    mol.thiele()
                yield prob, tuple(type(self)(mol) for mol in reaction.products)

    def __bool__(self):
        return self._molecule in self.__bb__


__all__ = ['RolloutSynthon']
