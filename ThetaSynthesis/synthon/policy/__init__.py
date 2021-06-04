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
from CGRtools import Reactor, SMILESRead
from io import TextIOWrapper
from itertools import takewhile
from torch import hstack, Tensor
from pickle import load
from pkg_resources import resource_stream
from StructureFingerprint import LinearFingerprint
from torch import from_numpy, sort
from .model import FilterNet, SorterNet, DoubleHeadedNet
from ..rollout import RolloutSynthon


class PolicySynthon(RolloutSynthon):
    __slots__ = ('_bit_string', )
    __filter__ = None
    __sorter__ = None
    __fragmentor__ = None

    __filter_path__ = None
    __sorter_path__ = None

    def __new__(cls, molecule, *args, **kwargs):
        if cls.__bb__ is None:
            with resource_stream(__name__, 'data/rules.pickle') as f:
                rules = load(f)
            cls.__bb__ = frozenset(str(m) for m in SMILESRead(TextIOWrapper(resource_stream(__name__, 'data/bb.smi'))))
            cls.__reactors__ = tuple(Reactor(x, delete_atoms=True) for x in rules)
            cls.__fragmentor__ = LinearFingerprint(length=4096, min_radius=2, max_radius=4, number_bit_pairs=4)
        if cls.__filter__ is None:
            cls.__filter__ = FilterNet().load_from_checkpoint(cls.__filter_path__)
            cls.__filter__.eval()
            cls.__sorter__ = SorterNet().load_from_checkpoint(cls.__sorter_path__)
            cls.__sorter__.eval()
        return super().__new__(cls, molecule, *args, **kwargs)

    def __init__(self, molecule, /):
        super().__init__(molecule)
        self._bit_string = from_numpy(self.__fragmentor__.transform([self._molecule])[0]).float()

    def _sorted(self):
        reactors = self.__reactors__
        vec = self._bit_string
        res = (self.__sorter__.forward(vec) + .01) * self.__filter__._predict(vec)
        sorted_, values = sort(res.sqrt(), descending=True)
        yield from ((x.item(), reactors[y.item()])
                    for x, y in zip(sorted_.squeeze(), values.squeeze())
                    if x > 0.)


class DoubleHeadedSynthon(PolicySynthon):
    __slots__ = ('_probs', )
    __net__ = None

    def __new__(cls, molecule, *args, **kwargs):
        if cls.__net__ is None:
            cls.__net__ = DoubleHeadedNet()
            cls.__net__.eval()
        return super().__new__(cls, molecule, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self._probs, float = self.__net__.forward(hstack([self._bit_string, Tensor([self._depth])]))
        self._float = float.item()

    def __float__(self):
        if self._float is not None:
            return self._float
        elif self:
            self._float = 1.
            return self._float

    def _sorted(self):
        reactors = self.__reactors__
        sorted_, values = sort(self._probs, descending=True)
        yield from ((x.item(), reactors[y.item()])
                    for x, y in takewhile(lambda x: x[0] > .1, zip(sorted_.squeeze(), values.squeeze())))


__all__ = ['PolicySynthon', 'DoubleHeadedSynthon']
