from abc import ABC, abstractmethod


class RetroTreeABC(ABC):
    __slots__ = ('_target', '_succ', '_pred', '_depth_stop', '_count_stop', '_terminal_count', '__dict__')

    @abstractmethod
    def __next__(self):
        """
        yield a path from target molecule to terminal node
        """

    def predecessor(self, node):
        return self._pred[node]

    def successors(self, node):
        return self._succ.get(node)


__all__ = ['RetroTreeABC']
