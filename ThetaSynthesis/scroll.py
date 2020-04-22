from abc import abstractmethod
from typing import overload, Iterable, MutableSequence

from . import Synthon
from .abc import ScrollABC


class Scroll(ScrollABC):
    def insert(self, index: int, object: Synthon) -> None:
        pass

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> Synthon: ...

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[Synthon]: ...

    def __getitem__(self, i: int) -> Synthon:
        pass

    @overload
    @abstractmethod
    def __setitem__(self, i: int, o: Synthon) -> None: ...

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[Synthon]) -> None: ...

    def __setitem__(self, i: int, o: Synthon) -> None:
        pass

    @overload
    @abstractmethod
    def __delitem__(self, i: int) -> None: ...

    @overload
    @abstractmethod
    def __delitem__(self, i: slice) -> None: ...

    def __delitem__(self, i: int) -> None:
        pass
