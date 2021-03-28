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
from itertools import filterfalse, tee
from typing import Tuple, Set
from .abc import ScrollABC
from .synthon.abc import SynthonABC


class Scroll(ScrollABC):
    __slots__ = ('_synthons', '_history', '_expand', '_closures', '_others', '_building_blocks')

    def __init__(self, synthons: Tuple[SynthonABC, ...], blocks: Set[SynthonABC],
                 history: Set[SynthonABC], others: Tuple['SynthonABC', ...], /):
        new_synths, new_blocks = self.partition(others)

        self._synthons = (*synthons, *new_synths)
        self._others = others

        self._building_blocks = blocks
        self._building_blocks |= set(new_blocks)

        self._history = history
        self._closures = set()  # expanded synthons available in history

        if self:
            self._expand = ()
        else:
            self._expand = iter(self._synthons[0])

    def __call__(self, **kwargs):
        for synth in self._others:
            synth(**kwargs)  # default scroll just transfer params into all new added synthons.

    def __bool__(self):
        """
        Is terminal state. All synthons is building blocks
        """
        return not self._synthons

    def __len__(self):
        return len(self._synthons)

    def __float__(self):
        """
        Worse value from all synthons in the scroll
        """
        return min(float(x) for x in self._synthons)

    @property
    def molecules(self):
        return tuple(x.molecule for x in self._synthons)

    @property
    def current_synthon(self):
        return self._synthons[0]

    @property
    def new_synthons(self):
        return self._others

    def __next__(self):
        """
        Expand Tree.
        """
        for prob, new in self._expand:
            if not self._history.isdisjoint(new):
                self._closures.add(new)
                continue
            blocks = self._building_blocks.copy()
            history = self._history.copy()
            history.update(new)
            return prob, type(self)((*self._synthons[1:], ), blocks, history, new)
        raise StopIteration('End of possible reactions has reached')

    def __hash__(self):
        return hash(tuple(hash(synth) for synth in self._synthons))

    def __repr__(self):
        return '\n'.join([repr(x) for x in self._synthons])

    @staticmethod
    def partition(iterable, key=bool):
        """
        Use a predicate to partition entries into false entries and true entries
        """
        t1, t2 = tee(iterable)
        return filterfalse(key, t1), filter(key, t2)


__all__ = ['Scroll']
