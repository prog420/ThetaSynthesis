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
from abc import ABC, abstractmethod
from CGRtools import MoleculeContainer, ReactionContainer
from typing import Dict, Set, Tuple


class ScrollABC(ABC):
    """
    Node of MCTS Tree
    """
    __slots__ = ()

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Tuple[float, 'ScrollABC']:
        """
        Yield pairs of reaction value and Scroll.
        """

    @abstractmethod
    def __bool__(self):
        ...

    @abstractmethod
    def __float__(self):
        ...

    @abstractmethod
    def __hash__(self):
        ...

    def __eq__(self, other: "ScrollABC"):
        return hash(self) == hash(other)

    @property
    @abstractmethod
    def molecules(self) -> Tuple[MoleculeContainer, ...]:
        """
        Molecules stored in scroll.
        """

    @abstractmethod
    def __call__(self, **kwargs):
        """
        Apply additional params from tree to scroll.

        Unified way for tree customizations.
        """


class RetroTreeABC(ABC):
    __slots__ = ('_succ', '_pred', '_nodes', '_free_node', '_visits', '_total_actions', '_probabilities')

    @abstractmethod
    def __init__(self, target: ScrollABC, /):
        self._succ: Dict[int, Set[int]] = {1: set()}
        self._pred: Dict[int, int] = {1: 0}
        self._nodes: Dict[int, ScrollABC] = {1: target}
        self._visits: Dict[int, int] = {1: 0}
        self._total_actions: Dict[int, float] = {1: 0.}
        self._probabilities: Dict[int, float] = {1: 0.}
        self._free_node: int = 2

    @abstractmethod
    def __next__(self) -> Tuple[ReactionContainer, ...]:
        """
        Yield a path from building blocks to target molecule.
        """

    def __iter__(self):
        return self

    def __len__(self):
        """
        Current size of tree
        """
        return self._free_node - 1

    def __repr__(self):
        return f'{type(self).__name__}({self._nodes[1]})'


__all__ = ['ScrollABC', 'RetroTreeABC']
