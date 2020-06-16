from math import sqrt
from typing import Dict
from .abc import RetroTreeABC
from . import Scroll, CombineSynthon

c_puct = 4


class RetroTree(RetroTreeABC):
    def __init__(self, target, class_name, stop_conditions: Dict):
        self._target = Scroll(synthons=tuple([class_name(target)]), reaction=None, probability=1., depth=0)
        self._succ = {self._target: set()}
        self._pred = {self._target: None}
        self._depth_stop = stop_conditions['depth_count']
        self._count_stop = stop_conditions['step_count']
        self._generator = self.__generator()

    def __next__(self):
        return next(self._generator)

    def __iter__(self):
        return self._generator

    def __generator(self):
        # FIXME too slow, need more speed
        # FIXME on next step of tree molecule must have less atoms
        # FIXME delete terminal stop
        if not self._count_stop:
            raise StopIteration
        else:
            max_count = len(self._succ) + self._count_stop
            while len(self._succ) <= max_count:
                print(len(self._succ))
                node = self._select()
                if node.depth == self._depth_stop:
                    self._backup(node, 0)
                if node:
                    break
                premolecules = node.premolecules
                for mol in premolecules:
                    self._pred[mol] = node
                self._succ[node] = set(premolecules)
                self._backup(node, node.value)
        terminal_nodes = [x for x in self._succ if x]
        if terminal_nodes:
            return self._path(terminal_nodes[0])
        else:
            return None

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
        while parent:
            scroll.increase_total_action(value)
            scroll.increase_visit_count()
            scroll = parent
            parent = self.predecessor(scroll)

    def _path(self, node: Scroll):
        """
        path from terminal node to root node
        """
        path = [node.get_reaction]
        while True:
            node = self._pred[node]
            try:
                path.append(node.get_reaction)
            except AttributeError:
                return path


__all__ = ['RetroTree']
