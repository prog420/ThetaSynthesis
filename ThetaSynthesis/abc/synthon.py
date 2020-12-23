from abc import ABC, abstractmethod
from CGRtools.containers import MoleculeContainer
from typing import Generator, Tuple


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

    @abstractmethod
    def value(self, depth: int) -> float:
        """
        value of molecule [-1; 1]
        """

    @abstractmethod
    def probabilities(self) -> Tuple[float, ...]:
        """
        vector of probabilities with len == premolecules
        """

    @abstractmethod
    def premolecules(self) -> Tuple[Tuple["Synthon", None, None], None, None]:
        """
        return tuple of tuples of Synthons (from nn)
        """


__all__ = ['SynthonABC']
