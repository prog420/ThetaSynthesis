from abc import ABC, abstractmethod
from CGRtools import MoleculeContainer
from typing import Dict, Optional, Set
from ..scroll import Scroll
from ..synthon import Synthon


class RetroTreeABC(ABC):
    __slots__ = ('_target', '_succ', '_pred', '_stop', '__dict__')

    def __init__(self, target: MoleculeContainer, stop_conditions: Dict):
        self._target = Synthon(target)
        self._succ: Dict[Scroll: Set[Scroll, ...]] = {self._target: set()}
        self._pred: Dict[Scroll: Optional[Scroll]] = {self._target: None}
        self._stop = stop_conditions

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """
        yield a path from target molecule to terminal node
        """

    def predecessor(self, node):
        return self._pred[node]

    def successors(self, node):
        return self._succ[node]


__all__ = ['RetroTreeABC']
