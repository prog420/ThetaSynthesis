from abc import ABC, abstractmethod


class RetroTreeABC(ABC):
    __slots__ = ('_target', '_succ', '_pred', '_depth_stop', '_count_stop', '_terminal_count', '__dict__')

    @abstractmethod
    def __next__(self):
        """
        yield a path from target molecule to terminal node
        """

    @abstractmethod
    def collect(self, number_positive: int, number_negative: int):
        """
        :param number_positive:
        :param number_negative:
        :return:
        """

    @abstractmethod
    def partial_fit(self):
        """
        :return:
        """

    def predecessor(self, node):
        return self._pred[node]

    def successors(self, node):
        return self._succ.get(node)


__all__ = ['RetroTreeABC']
