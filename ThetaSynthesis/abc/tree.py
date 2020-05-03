from abc import ABC, abstractmethod
from CGRtools import MoleculeContainer
from typing import Dict, Optional, Set
from ..scroll import Scroll
from ..synthon import Synthon, CombineSynthon, SlowSynthon, StupidSynthon


class RetroTreeABC(ABC):
    __slots__ = ('_target', '_succ', '_pred', '_mol_depot')

    def __init__(self, target: MoleculeContainer, stop_conditions: Dict, wrapper: str):
        if wrapper == 'combine':
            self._target = CombineSynthon(target)
        elif wrapper == 'slow':
            self._target = SlowSynthon(target)
        elif wrapper == 'stupid':
            self._target = StupidSynthon(target)
        else:
            raise TypeError
        self._succ: Dict[Scroll: Set[Scroll, ...]] = {self._target: set()}
        self._pred: Dict[Scroll: Optional[Scroll]] = {self._target: None}
        self._mol_depot: Dict[MoleculeContainer: Synthon] = {target: self._target}

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """
        yield a path from target molecule to terminal node
        """

    @abstractmethod
    def _select(self):
        """
        return a node which have not expanded yet and will be expanded now
        """

    def predecessor(self, node):
        return self._pred[node]

    def successors(self, node):
        return self._succ[node]


__all__ = ['RetroTreeABC']
