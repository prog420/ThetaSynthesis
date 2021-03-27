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
from CGRtools import MoleculeContainer, ReactionContainer
from math import sqrt
from tqdm import tqdm
from typing import Type, Tuple, Optional
from .abc import RetroTreeABC
from .scroll import Scroll
from .synthon.abc import SynthonABC


class RetroTree(RetroTreeABC):
    __slots__ = ('_depth', '_size', '_c_puct', '_expanded', '_iterations', '_limit', '_found', '_tqdm', '_node_depth')

    def __init__(self, target: MoleculeContainer, /, synthon_class: Type[SynthonABC],
                 c_puct: float = 4., depth: int = 10, size: int = 1e4, iterations: int = 1e6):
        """
        :param target: target molecule
        :param c_puct: breadth/depth criterion
        :param depth: max length of path to building blocks
        :param size: max size of tree
        :param iterations: limit of iterations
        """
        self._depth = depth
        self._size = int(size)
        self._c_puct = c_puct
        self._expanded = 1
        self._limit = iterations = int(iterations)
        self._iterations = 0
        self._found = 0
        self._node_depth = {1: 0}
        self._tqdm = tqdm(total=iterations)

        synthon = synthon_class(target)
        scroll = Scroll((synthon, ), {synthon}, 1)
        scroll(finish=self._depth)
        super().__init__(scroll)

    def __del__(self):
        self._tqdm.close()

    def _add(self, node: int, scroll: Scroll, prob: float):
        """
        Add new node to tree.
        """
        new_node = self._free_node
        self._nodes[new_node] = scroll
        self._pred[new_node] = node
        self._succ[node].add(new_node)
        self._succ[new_node] = set()
        self._visits[new_node] = 0
        self._probabilities[new_node] = prob
        self._total_actions[new_node] = 0.
        self._node_depth[new_node] = self._node_depth[node] + 1
        self._free_node += 1

    def _update_visits(self, node: int):
        """
        Increment visits count in path from given node to root.
        """
        while node:
            self._visits[node] += 1
            node = self._pred[node]

    def _update_actions(self, node: int):
        """
        Update total action of each node in path to root by value of given node.
        """
        value = float(self._nodes[node])
        while node:
            self._total_actions[node] += value
            node = self._pred[node]

    def _expand(self, node: int):
        """
        Expand new node.
        """
        finish = self._depth - self._node_depth[node]
        for prob, scroll in self._nodes[node]:
            scroll(finish=finish)  # init new scroll
            self._add(node, scroll, prob)

    def _select(self, node: int) -> int:
        """
        Select preferred successor node based on views count and synthesisability.
        """
        return max(self._succ[node], key=self._puct)

    def _puct(self, node):
        """
        Polynomial upper confidence trees criterion for choosing node from tree
        """
        prob = self._probabilities[node]
        visit = self._visits[node]

        # C_PUCT is a constant determining a level of exploration; can be from 1 to 6; 4 is more balanced value
        u = self._c_puct * prob * sqrt(sum(self._visits[x] + 1 for x in self._succ[self._pred[node]]))

        return (self._total_actions[node] + u) / (visit + 1)

    def _prepare_path(self, node: int) -> Tuple[ReactionContainer, ...]:
        """
        Prepare reaction path

        :param node: building block node
        """
        nodes = []
        while node:
            nodes.append(node)
            node = self._pred[node]

        tmp = []
        for node in reversed(nodes):
            node = self._nodes[node]
            tmp.append(node.molecules)
        tmp = [ReactionContainer(after[len(before) - 1:], [before[0].copy()]) for before, after in zip(tmp, tmp[1:])]
        for r in tmp:
            r.fix_positions()
        return tuple(reversed(tmp))

    def __next__(self):
        while self._expanded < self._free_node:
            self._iterations += 1
            if self._iterations > self._limit:
                raise StopIteration('Iterations limit exceeded. \n' + self.report())
            self._tqdm.update()
            depth = 0
            node = 1
            while True:
                if self._visits[node]:  # already expanded
                    if not self._succ[node]:  # dead terminal non-building block node.
                        self._update_visits(node)
                        break
                    node = self._select(node)
                    depth += 1
                else:
                    self._expanded += 1  # increment visited nodes count.
                    if self._nodes[node]:  # found path!
                        self._update_visits(node)  # this prevents expanding of bb node
                        # I dunno: self._update_actions(node)
                        self._found += 1
                        return self._prepare_path(node)
                    elif depth < self._depth and self._free_node < self._size:  # expand if depth limit not reached
                        self._expand(node)
                        self._update_visits(node)  # mark node as visited
                        self._update_actions(node)
                        break
                    else:
                        self._update_visits(node)
                        self._update_actions(node)
                        break
        raise StopIteration('Max tree size exceeded or all possible paths found' + self.report())

    def report(self):
        return f'Tree for: {self._nodes[1]}\n' \
               f'Size: {len(self)}\nNumber of unvisited nodes: {self._free_node - self._expanded}\n' \
               f'Found paths: {self._found}'

    def visualize(self, draw_format: str = 'png', prog: Optional[str] = None,
                  only_visited: bool = False, verbose: int = 2):
        import pygraphviz as pgv
        from warnings import warn

        lst = ['id', 'visits', 'smiles in queue', 'value']
        if verbose == 3 and not only_visited:
            warn('Verbose = 3 option can be used only for visited nodes', UserWarning)
            only_visited = True
        elif verbose == 2:
            lst = lst[:3]
        elif verbose == 1:
            lst = lst[:2]
        elif verbose == 0:
            lst = lst[:1]

        if only_visited:
            nodes = {k: v for k, v in self._nodes.items() if self._visits[k]}
        else:
            nodes = {k: v for k, v in self._nodes.items()}

        lambdas = [lambda x: x, lambda x: self._visits[x], lambda x: repr(x), lambda x: float(self._nodes[x])]

        nodes_with_attrs = {
            k: '\n'.join(f'{x}: {z(y)}' for x, y, z in zip(lst, [k, k, v, k], lambdas))
            for k, v
            in nodes.items()
        }
        pred = self._pred
        g = pgv.AGraph(directed=True)
        g.node_attr['shape'] = 'box'
        g.add_edges_from([(v, k) for k, v in pred.items() if k != 1 and k in nodes])

        for k, node in zip(nodes, g.nodes()):
            node.attr['label'] = nodes_with_attrs[k]

        g.layout(prog='dot')
        return g.draw(format=draw_format, prog=prog)


__all__ = ['RetroTree']
