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
from CGRtools import MoleculeContainer
from typing import Type
from .abc import RetroTreeABC, SynthonABC
from .scroll import Scroll


class RetroTree(RetroTreeABC):
    __slots__ = ()

    def __init__(self, target: MoleculeContainer, /, synthon_class: Type[SynthonABC], depth=10, size=10000):
        synthon = synthon_class(target)
        super().__init__(Scroll((synthon,), {synthon}), depth=depth, size=size)

    def _add(self, pred: int, node: Scroll):
        """
        Add new node to tree.
        """
        succ = self._free_node
        self._nodes[succ] = node
        self._pred[succ] = pred
        self._succ[pred].add(succ)
        self._succ[succ] = set()
        self._visits[succ] = 1
        self._free_node += 1

    def _update(self, pred: int):
        """
        Increment visits count in path from given node to root.
        """
        visits = self._visits
        preds = self._pred
        while pred:
            visits[pred] += 1
            pred = preds[pred]

    def __next__(self):
        pass


__all__ = ['RetroTree']
