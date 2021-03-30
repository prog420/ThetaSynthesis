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
from .abc import ScrollABC, IsTerminal
from .synthon.abc import SynthonABC


class Scroll(ScrollABC):
    __slots__ = ('_synthons', '_history', '_expand', '_closures', '_new_synthons')

    def __init__(self, synthons: Tuple[SynthonABC, ...], new_synthons: Tuple[SynthonABC, ...],
                 history: Set[SynthonABC], /):
        self._synthons = (*synthons, *(x for x in new_synthons if not x))
        self._new_synthons = new_synthons
        self._history = history
        self._closures = set()  # expanded synthons available in history

        if not self._synthons:
            self._expand = ()
        else:
            self._expand = iter(self._synthons[0])

    def __call__(self, **kwargs):
        for synth in self._new_synthons:
            synth(**kwargs)  # default scroll just transfer params into all new added synthons.

    @property
    def current_synthon(self):
        try:
            return self._synthons[0]
        except IndexError:
            raise IsTerminal

    @property
    def new_synthons(self):
        return self._new_synthons

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
        return min((float(x) for x in self._synthons), default=1.)

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
            return prob, type(self)(self._synthons[1:], new, history)
        raise StopIteration('End of possible reactions has reached')

    def __hash__(self):
        return hash(tuple(hash(synth) for synth in self._synthons))

    def __repr__(self):
        s = '\n'.join([repr(x) for x in self._synthons])
        n = '\n'.join([repr(x) for x in self._new_synthons])
        return f'queue:\n{s}\nnew:\n{n}'


__all__ = ['Scroll']
