from math import sqrt
from typing import Dict, Optional
from .abc import RetroTreeABC
from .scroll import Scroll
from .synthon import CombineSynthon

C_PUCT = 4


class RetroTree(RetroTreeABC):
    def __init__(self, target, class_name=CombineSynthon, stop_conditions: Optional[Dict] = None):
        self._target = Scroll(synthons=tuple([class_name(target)]), reaction=None, probability=1., depth=0)
        self._succ = {self._target: set()}
        self._pred = {self._target: None}

        if stop_conditions is None:
            self._depth_stop = 10
            self._count_stop = 10000
        else:
            self._depth_stop = stop_conditions['depth_count']
            self._count_stop = stop_conditions['step_count']

        self._generator = self.__generator()

        self._successful = set()
        self._fitted = False

    def __next__(self):
        return next(self._generator)

    def __iter__(self):
        return self._generator

    def __generator(self):
        max_count = self._count_stop
        while len(self._succ) < max_count:
            node = self._select()
            if node.depth == self._depth_stop and not node:
                self._backup(node, -1)
                continue
            elif node:
                self._backup(node, 1)
                self._successful.add(node)

                yield self._path(node)
                continue
            premolecules = list(node.premolecules())
            for mol in premolecules:
                self._pred[mol] = node
            self._succ[node] = set(premolecules)
            self._backup(node, node.value)
        self._fitted = True

    @property
    def fitted(self):
        return self._fitted

    def _puct(self, scroll: Scroll) -> float:
        mean_action = scroll.mean_action
        visit_count = scroll.visit_count
        probability = scroll.probability
        summary_visit_count = sum(node.visit_count for node in self._comrades(scroll))
        ucp = C_PUCT * probability * (sqrt(summary_visit_count) / (1 + visit_count))
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
            if node.reaction is not None:
                path.append(node.reaction)
            else:
                return path

    def generate_examples(self):
        if not self.fitted:
            _ = list(self)
        yield from (((node, 1), node.depth) for node in self._successful)
        yield from (((node, -1), node.depth) for node in sorted(
            {node for node in self._succ if not self._succ.get(node)},
            key=lambda x: x.visit_count
        ))


__all__ = ['RetroTree']
