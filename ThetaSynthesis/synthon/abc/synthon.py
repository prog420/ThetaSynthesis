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
from abc import ABC, abstractmethod
from CGRtools import MoleculeContainer
from typing import Tuple, Iterator


class SynthonABC(ABC):
    __slots__ = ('_molecule', '_mapping')
    __cache__ = {}

    def __init__(self, molecule: MoleculeContainer, /):
        if str(molecule) in self.__cache__:
            self._molecule = self.__cache__[str(molecule)]
            self._mapping = self._molecule.get_fast_mapping(molecule)
        else:
            self.__cache__[str(molecule)] = molecule
            self._molecule = molecule
            self._mapping = None

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[float, Tuple['SynthonABC', ...]]]:
        """
        Generator of precursors synthons. Ordered by preference of reactions.

        :return: Pairs of reaction value and tuple of synthons.
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
        if self._mapping:
            return self._molecule.remap(self._mapping, copy=True)
        return self._molecule.copy()

    def __repr__(self):
        return str(self._molecule)


__all__ = ['SynthonABC']