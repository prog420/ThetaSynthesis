from collections import deque
from math import sqrt
from typing import Dict
from .abc import RetroTreeABC
from . import Scroll, CombineSynthon

c_puct = 4


class RetroTree(RetroTreeABC):
    # TODO return positive and negative examples
    # TODO add partial_fit of tree
    def __init__(self, target, class_name, stop_conditions: Dict, generate_false=False):
        self._target = Scroll(synthons=tuple([class_name(target)]), reaction=None, probability=1., depth=0)
        self._succ = {self._target: set()}
        self._pred = {self._target: None}
        self._depth_stop = stop_conditions['depth_count']
        self._count_stop = stop_conditions['step_count']
        self._generate_false = generate_false
        self._generator = self.__generator()

    def __next__(self):
        return next(self._generator)

    def __iter__(self):
        return self._generator

    def __generator(self):
        max_count = self._count_stop
        while len(self._succ) < max_count:
            print(len(self._succ))
            node = self._select()
            if node.depth == self._depth_stop and not node:
                self._backup(node, -1)
                continue
            elif node:
                self._backup(node, 1)
                yield self._path(node)
                continue
            premolecules = list(node.premolecules())
            for mol in premolecules:
                self._pred[mol] = node
            self._succ[node] = set(premolecules)
            self._backup(node, node.value)

    def _puct(self, scroll: Scroll) -> float:
        mean_action = scroll.mean_action
        visit_count = scroll.visit_count
        probability = scroll.probability
        summary_visit_count = sum(node.visit_count for node in self._comrades(scroll))
        ucp = c_puct * probability * (sqrt(summary_visit_count) / (1 + visit_count))
        return mean_action + ucp

    def _select(self) -> Scroll:
        """
        select node with best q+u and which haven't expanded yet
        """
        scroll = self._target
        children = self.successors(scroll)
        while children:
            scroll = max(children, key=self._puct)
            children = self.successors(scroll)
        return scroll

    def _comrades(self, scroll: Scroll):
        """
        nodes on the same step of tree with the same predecessor (neighbors)
        """
        return self.successors(self.predecessor(scroll))

    def _backup(self, scroll: Scroll, value: float):
        """
        increment to total action value for each node in this branch and
        increment visit count on 1
        if node is terminal then value = 0
        """
        parent = self.predecessor(scroll)
        while parent is not None:
            scroll.increase_total_action(value)
            scroll.increase_visit_count()
            scroll = parent
            parent = self.predecessor(scroll)

    def _path(self, node: Scroll):
        """
        path from terminal node to root node
        """
        path = [node.reaction]
        while True:
            node = self._pred[node]
            try:
                path.append(node.reaction)
            except AttributeError:
                return path

    def __worse_first(self):
        stack = deque([(self._target, self._target.depth)])
        path = []
        while stack:
            current, depth = stack.pop()
            path = path[:depth]
            path.append(current)
            depth += 1
            if current not in self._succ:
                yield path
            else:
                succs = [(x, depth) for x in sorted(self._succ[current], key=lambda x: x.visit_count)]
                stack.extend(succs)


__all__ = ['RetroTree']
