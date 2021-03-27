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
from typing import Tuple, Set
from .abc import ScrollABC
from .synthon.abc import SynthonABC


class Scroll(ScrollABC):
    __slots__ = ('_synthons', '_history', '_expand', '_closures', '_others')

    def __init__(self, synthons: Tuple[SynthonABC, ...], history: Set[SynthonABC], others: int, /):
        self._synthons = synthons
        self._others = others
        self._history = history
        self._closures = set()  # expanded synthons available in history

        current = synthons[0]
        if current:  # already building block
            self._expand = ()
        else:
            self._expand = iter(current)

    def __call__(self, **kwargs):
        for synth in self._synthons[-self._others:]:
            synth(**kwargs)  # default scroll just transfer params into all new added synthons.

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
    def molecules(self):
        return tuple(x.molecule for x in self._synthons)

    def __next__(self):
        """
        Expand Tree.
        """
        for prob, new in self._expand:
            if not self._history.isdisjoint(new):
                self._closures.add(new)
                continue
            history = self._history.copy()
            history.update(new)
            return prob, type(self)((*self._synthons[1:], *sorted(new, key=bool)), history, len(new))
        raise StopIteration('End of possible reactions has reached')

    def __hash__(self):
        return hash(tuple(hash(synth) for synth in self._synthons))

    def __repr__(self):
        return '\n'.join([repr(x) for x in self._synthons])


__all__ = ['Scroll']
