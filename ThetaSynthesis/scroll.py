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
from typing import Tuple, Set
from .abc import SynthonABC, ScrollABC


class Scroll(ScrollABC):
    __slots__ = ('_synthons', '_current', '_history', '_expand', '_closures', '_others')

    def __init__(self, synthons: Tuple[SynthonABC, ...], history: Set[SynthonABC], /):
        self._synthons = synthons
        self._current = synthons[0]
        self._others = synthons[1:]
        self._history = history
        self._closures = set()  # expanded synthons available in history
        self._expand = iter(synthons[0])

    def __bool__(self):
        """
        Is terminal state. All synthons is building blocks
        """
        return all(self._synthons)

    def __len__(self):
        return len(self._synthons)

    def __float__(self):
        """
        Worse value from all synthons in the scroll
        """
        return min(float(x) for x in self._synthons)

    @property
    def current(self):
        return self._current

    def __next__(self) -> 'Scroll':
        """
        Expand Tree.
        """
        history = self._history
        for new in self._expand:
            if not history.isdisjoint(new):
                self._closures.add(new)
                continue
            history = self._history.copy()
            history.update(new)
            return type(self)((*self._others, *new), history)
        raise StopIteration('End of possible reactions has reached')

    def __iter__(self):
        return self


__all__ = ['Scroll']
