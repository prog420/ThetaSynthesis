# -*- coding: utf-8 -*-
#
#  Copyright 2020-2021 Alexander Sizov <>
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
from abc import ABCMeta, abstractmethod
from CGRtools import MoleculeContainer
from typing import Tuple, Iterator


class SynthonABCMeta(ABCMeta):
    __singletons__ = {}

    def __call__(cls, molecule: MoleculeContainer):
        try:
            return cls.__singletons__[molecule]
        except KeyError:
            cls.__singletons__[molecule] = st = super().__call__(molecule)
            return st


class SynthonABC(metaclass=SynthonABCMeta):
    __slots__ = ('_molecule',)

    def __init__(self, molecule: MoleculeContainer, /):
        self._molecule = molecule

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple['SynthonABC', ...]]:
        """
        Generator of precursors synthons.
        """

    @abstractmethod
    def __bool__(self):
        """
        Is building block.
        """

    @abstractmethod
    def __float__(self):
        """
        Value of synthesisability.
        """

    def __hash__(self):
        return hash(self._molecule)

    def __eq__(self, other: 'SynthonABC'):
        return self._molecule == other._molecule

    @property
    def molecule(self) -> MoleculeContainer:
        return self._molecule.copy()


__all__ = ['SynthonABC']
