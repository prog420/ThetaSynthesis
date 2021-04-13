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
from CGRtools import Reactor, SMILESRead, smiles
from io import TextIOWrapper
from itertools import takewhile
from pickle import load
from pkg_resources import resource_stream
from StructureFingerprint import LinearFingerprint
from torch import from_numpy, sort
from .model import JustPolicyNet
from ..rollout import RolloutSynthon


class PolicySynthon(RolloutSynthon):
    __slots__ = ('_bit_string', )
    __net__ = None
    __fragmentor__ = None

    def __new__(cls, molecule, *args, **kwargs):
        if cls.__bb__ is None:
            with resource_stream(__name__, 'data/rules_reverse.pickle') as f:
                rules = load(f)
            cls.__bb__ = frozenset(str(m) for m in SMILESRead(TextIOWrapper(resource_stream(__name__, 'data/bb.smi'))))
            cls.__reactors__ = tuple(Reactor(x, delete_atoms=True) for x in rules)

            cls.__net__ = JustPolicyNet()
            cls.__net__.eval()
            cls.__fragmentor__ = LinearFingerprint(length=4096, min_radius=2, max_radius=4, number_bit_pairs=4)
        return super().__new__(cls, molecule, *args, **kwargs)

    def __init__(self, molecule, /):
        super().__init__(molecule)
        self._bit_string = from_numpy(self.__fragmentor__.transform([self._molecule])).float()

    def _sorted(self):
        reactors = self.__reactors__
        sorted_, values = sort(self.__net__.forward(self._bit_string), descending=True)
        yield from ((x.item(), reactors[y.item()]) for x, y in takewhile(lambda x: x[0] > .1, zip(sorted_.squeeze(), values.squeeze())))


__all__ = ['PolicySynthon']
