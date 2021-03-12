from abc import ABC, abstractmethod
from CGRtools import ReactionContainer
from typing import Dict, Set, Tuple


class ScrollABC(ABC):
    """
    Node of MCTS Tree
    """
    __slots__ = ()


class RetroTreeABC(ABC):
    __slots__ = ('_target', '_succ', '_pred', '_nodes', '_depth', '_size', '_free_node')

    @abstractmethod
    def __init__(self, target: ScrollABC, /, depth: int = 10, size: int = 10000):
        """
        :param target: target molecule
        :param depth: max path to building blocks
        :param size: max size of tree
        """
        self._target = target
        self._succ: Dict[int, Set[int]] = {1: set()}
        self._pred: Dict[int, int] = {1: 1}
        self._nodes: Dict[int, ScrollABC] = {1: target}
        self._depth = depth
        self._size = size
        self._free_node: int = 1

    @abstractmethod
    def __next__(self) -> Tuple[ReactionContainer, ...]:
        """
        Yield a path from target molecule to building blocks.
        """

    def __iter__(self):
        return self

    def __len__(self):
        """
        Current size of tree
        """
        return self._free_node


__all__ = ['ScrollABC', 'RetroTreeABC']
