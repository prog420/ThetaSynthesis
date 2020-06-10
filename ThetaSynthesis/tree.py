from math import sqrt
from typing import Dict
from .abc import RetroTreeABC
from . import Scroll, CombineSynthon

c_puct = 4


class RetroTree(RetroTreeABC):
    def __init__(self, target, class_name, stop_conditions: Dict):
        self._target = Scroll(synthons=tuple([class_name(target)]), reaction=None, probability=1., depth=0)
        self._succ = {self._target: {}}
        self._pred = {self._target: None}
        self._depth_stop = stop_conditions['depth_count']
        self._count_stop = stop_conditions['step_count']
        self._terminal_stop = stop_conditions['terminal_count']

    def __next__(self):
        # FIXME too slow, need more speed
        # FIXME on next step of tree molecule must have less atoms
        while self._count_stop and self._terminal_stop:
            print(self._count_stop)
            node = self._select
            if node.depth == self._depth_stop or node:
                self._backup(node, 0)
                self._terminal_stop -= 1
                yield self._path(node)
            premolecules = node.premolecules
            for mol in premolecules:
                self._pred[mol] = node
            self._succ[node] = set(premolecules)
            self._backup(node, node.value)
            self._count_stop -= 1
        raise StopIteration

    def _puct(self, scroll: Scroll) -> float:
        mean_action = scroll.mean_action
        visit_count = scroll.visit_count
        probability = scroll.probability
        summary_visit_count = sum([node.visit_count for node in self._comrades(scroll)])
        ucp = c_puct * probability * (sqrt(summary_visit_count) / (1 + visit_count))
        return mean_action + ucp

    @property
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
        return self.successors(self.predecessor(scroll))

    def _backup(self, scroll: Scroll, value: float):
        """
        increment to total action value for each node in this branch and
        increment visit count on 1
        if node is terminal then value = 0
        """
        parent = self.predecessor(scroll)
        while parent:
            scroll.increase_total_action(value)
            scroll.increase_visit_count()
            scroll = parent
            parent = self.predecessor(scroll)

    def _path(self, node: Scroll):
        path = [node.get_reaction]
        while True:
            node = self._pred[node]
            try:
                path.append(node.get_reaction)
            except AttributeError:
                return path


__all__ = ['RetroTree']
