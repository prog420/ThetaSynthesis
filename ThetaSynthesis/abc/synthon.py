from abc import ABCMeta, abstractmethod
from CGRtools import MoleculeContainer
from typing import Tuple, Iterator


class SynthonABCMeta(ABCMeta):
    __singletons__ = {}

    def __call__(cls, molecule: MoleculeContainer):
        try:
            return cls.__singletons__[molecule]
        except KeyError:
            cls.__singletons__[molecule] = st = super().__call__(molecule)
            return st


class SynthonABC(metaclass=SynthonABCMeta):
    __slots__ = ('_molecule',)

    def __init__(self, molecule: MoleculeContainer, /):
        self._molecule = molecule

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple['SynthonABC', ...]]:
        """
        Generator of precursors synthons.
        """

    @abstractmethod
    def __bool__(self):
        """
        Is building block.
        """

    @abstractmethod
    def __float__(self):
        """
        Value of synthesisability.
        """

    def __hash__(self):
        return hash(self._molecule)

    def __eq__(self, other: 'SynthonABC'):
        return self._molecule == other._molecule


__all__ = ['SynthonABC']
