from abc import ABC, abstractmethod
from CGRtools.containers import MoleculeContainer
from typing import List, Tuple


class SynthonABC(ABC):
    __slots__ = ('_molecule', '__dict__')
    __singletone__ = {}

    def __new__(cls, molecule: MoleculeContainer):
        if molecule in cls.__singletone__:
            return cls.__singletone__[molecule]
        else:
            obj = object.__new__(cls)
            cls.__singletone__[molecule] = obj
            obj._molecule = molecule
            return obj

    @property
    @abstractmethod
    def value(self) -> float:
        """
        value of molecule [-1; 1]
        """

    @property
    @abstractmethod
    def probabilities(self) -> Tuple[float, ...]:
        """
        vector od probalities with len == premolecules
        """

    @property
    @abstractmethod
    def premolecules(self) -> Tuple[Tuple['SynthonABC', ...], ...]:
        """
        return tuple of tuples os Synthons (from nn)
        """


__all__ = ['SynthonABC']
