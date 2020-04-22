from abc import abstractmethod
from collections.abc import MutableSequence
from typing import overload, Iterable
from ..synthon import Synthon


class ScrollABC(MutableSequence):
    """
    An implementation of an queue in nodes
    I did not come up with anything better than call this class like "scroll"
    I am not sure about inherit from mutable sequence, think about deque as well
    """
    @abstractmethod
    def insert(self, index: int, object: Synthon) -> None: ...

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> Synthon: ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[Synthon]: ...

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: Synthon) -> None: ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[Synthon]) -> None: ...

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None: ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None: ...

    def __len__(self) -> int:
        pass
