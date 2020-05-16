from math import sqrt
from .abc import RetroTreeABC
from .scroll import Scroll

c_puct = 4


class RetroTree(RetroTreeABC):
    def __next__(self):
        """
        return one path from root node to terminal node
        """
        yield

    def _puct(self, scroll):
        mean_action = scroll.mean_action
        visit_count = scroll.visit_count
        probability = scroll.probability
        summary_visit_count = sum([node.visit_count for node in self._comrades(scroll)])
        ucp = c_puct * probability * (sqrt(summary_visit_count) / (1 + visit_count))
        return mean_action + ucp

    @property
    def _select(self):
        """
        select node with best q+u and which haven't expanded yet
        """
        scroll = self._target
        children = self.successors(scroll)
        while children:
            scroll = max(children, key=self._puct)
            children = self.successors(scroll)
        return scroll

    def _comrades(self, scroll):
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


__all__ = ['RetroTree']
