from abc import ABC, abstractmethod
from CGRtools import MoleculeContainer, ReactionContainer
from typing import Dict, Set, Tuple


class ScrollABC(ABC):
    """
    Node of MCTS Tree
    """
    __slots__ = ()


class RetroTreeABC(ABC):
    __slots__ = ('_target', '_succ', '_pred', '_nodes', '_depth', '_size')

    @abstractmethod
    def __init__(self, target: MoleculeContainer, /, depth: int = 10, size: int = 10000):
        """
        :param target: target molecule
        :param depth: max path to building blocks
        :param size: max size of tree
        """
        self._succ: Dict[int, Set[int]]
        self._pred: Dict[int, int]
        self._nodes: Dict[int, ScrollABC]
        self._target: ScrollABC
        self._depth = depth
        self._size = size

    @abstractmethod
    def __next__(self) -> Tuple[ReactionContainer, ...]:
        """
        Yield a path from target molecule to building blocks.
        """

    def __iter__(self):
        return self

    @abstractmethod
    def __len__(self):
        """
        Current size of tree
        """


__all__ = ['ScrollABC', 'RetroTreeABC']
