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
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..synthon.abc import SynthonABC


class IsTerminal(Exception):
    """Scroll is terminal"""


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

    def __eq__(self, other: 'ScrollABC'):
        return hash(self) == hash(other)

    @property
    @abstractmethod
    def current_synthon(self) -> 'SynthonABC':
        """
        Return a synthon from the top of the queue of node's synthons.
        raise IsTerminal exception then scroll do not contains non-building blocks.
        """

    @property
    @abstractmethod
    def new_synthons(self) -> Tuple['SynthonABC', ...]:
        """
        Return a collection of new just added synthons.
        """

    @abstractmethod
    def __call__(self, **kwargs):
        """
        Apply additional params from tree to scroll.

        Unified way for tree customizations.
        """


__all__ = ['ScrollABC', 'IsTerminal']
