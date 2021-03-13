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
from CGRtools import MoleculeContainer, ReactionContainer
from typing import Type, Tuple
from .abc import RetroTreeABC, SynthonABC
from .scroll import Scroll


class RetroTree(RetroTreeABC):
    __slots__ = ('_depth', '_size')

    def __init__(self, target: MoleculeContainer, /, synthon_class: Type[SynthonABC], depth: int = 10, size: int = 1e4):
        """
        :param target: target molecule
        :param depth: max path to building blocks
        :param size: max size of tree
        """
        synthon = synthon_class(target)
        self._depth = depth
        self._size = size
        super().__init__(Scroll((synthon,), {synthon}))

    def _add(self, node: int, scroll: Scroll):
        """
        Add new node to tree.
        """
        new_node = self._free_node
        self._nodes[new_node] = scroll
        self._pred[new_node] = node
        self._succ[node].add(new_node)
        self._succ[new_node] = set()
        self._visits[new_node] = 0
        self._free_node += 1

    def _update(self, node: int):
        """
        Increment visits count in path from given node to root.
        """
        visits = self._visits
        preds = self._pred
        while node:
            visits[node] += 1
            node = preds[node]

    def _expand(self, node: int):
        """
        Expand new node.
        """
        for scroll in self._nodes[node]:
            self._add(node, scroll)

    def _select(self, node: int) -> int:
        """
        Select preferred successor node based on views count and synthesisability.
        """
        # todo: implement.

    def _prepare_path(self, node: int) -> Tuple[ReactionContainer, ...]:
        """
        Prepare reaction path

        :param node: building block node
        """
        preds = self._pred
        nodes = [node]
        while node:
            node = preds[node]
            nodes.append(node)

        scrolls = self._nodes
        tmp = []
        for node in reversed(nodes):
            node = scrolls[node]
            tmp.append(node.molecules)
        tmp = [ReactionContainer(after[len(before) - 1:], before[:1]) for before, after in zip(tmp, tmp[1:])]
        return tuple(reversed(tmp))

    def __next__(self):
        while self._free_node <= self._size:
            depth = 0
            node = 1
            while True:
                if self._visits[node]:  # already expanded
                    if not self._succ[node]:  # dead terminal non-building block node.
                        self._update(node)
                        break
                    node = self._select(node)
                    depth += 1
                elif self._nodes[node]:  # found path!
                    self._update(node)  # this prevents expanding of bb node
                    return self._prepare_path(node)
                elif depth < self._depth:  # expand if depth limit not reached
                    self._update(node)  # mark node as visited
                    self._expand(node)
                    break


__all__ = ['RetroTree']
