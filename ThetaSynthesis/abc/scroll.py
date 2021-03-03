from abc import abstractmethod, ABC
from functools import cached_property
from CGRtools.containers import ReactionContainer
from typing import Optional, Tuple, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from CGRtools.containers import MoleculeContainer


class ScrollABC(ABC):
    __slots__ = ('_synthons', '_reaction', '_depth', '_visit_count', '_value', '_total_action', '_probability',
                 '_rule_number', '__dict__')

    def __init__(self, synthons, reaction: Optional[ReactionContainer], probability: float, depth: int,
                 rule_number: Optional[int] = None, parents: Optional[Set["MoleculeContainer"]] = None):
        if parents is None:
            parents = set()
        self._synthons = synthons
        self._reaction = reaction
        self._depth = depth
        self._visit_count = 0
        self._total_action = 0.
        self._value = self.worse_value
        self._probability = probability
        self._parents = parents
        self._rule_number = rule_number

    def __bool__(self):
        """
        True if a node is terminal
        """
        return not self._synthons or bool(self._meet_again)

    def __len__(self):
        return len(self._synthons)

    @property
    def _meet_again(self):
        return {x.molecule for x in self._synthons} & self._parents

    @abstractmethod
    def premolecules(self) -> Tuple['ScrollABC', ...]:
        """
        successors nodes
        """

    @property
    @abstractmethod
    def worse_value(self):
        """
        worse value from all synthons in the scroll
        """

    @property
    def target(self):
        return self._synthons[0]

    @cached_property
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

    @property
    def value(self):
        return self._value

    @property
    def reaction(self):
        return self._reaction

    @property
    def probability(self):
        return self._probability

    @property
    def rule_number(self):
        return self._rule_number


__all__ = ['ScrollABC']
