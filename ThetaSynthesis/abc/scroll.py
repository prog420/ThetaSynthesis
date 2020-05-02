from abc import abstractmethod, ABC
from CGRtools.containers import ReactionContainer
from typing import Tuple
from ..synthon import Synthon


class ScrollABC(ABC):
    __slots__ = ('_synthons', '_reaction', '_depth', '_visit_count', '_total_action', '__dict__')

    def __init__(self, synthons: Tuple[Synthon, ...], reaction: ReactionContainer, depth: int):
        self._synthons = synthons
        self._reaction = reaction
        self._depth = depth
        self._visit_count = 0
        self._total_action = 0.

    def __bool__(self):
        return not self._synthons

    @property
    @abstractmethod
    def premolecules(self) -> Tuple['ScrollABC', ...]:
        """
        succesors nodes
        """

    @property
    @abstractmethod
    def worse_value(self):
        """
        worse value from all synthons in the scroll
        """

    @property
    def mean_action(self):
        if not self._visit_count:
            return 0.
        return self._total_action / self._visit_count

    def increase_visit_count(self):
        self._visit_count += 1

    def increase_total_action(self, value: float):
        self._total_action += value

    @property
    def depth(self):
        return self._depth

    @property
    def visit_count(self):
        return self._visit_count


__all__ = ['ScrollABC']
