from abc import ABC, abstractmethod
from CGRtools.containers import MoleculeContainer
from typing import Tuple


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
        Yield a tuple with sorted probabilities for each rules
        """

    @abstractmethod
    def premolecules(self) -> Tuple[Tuple["Synthon", None, None], None, None]:
        """
        Yield tuple with tuples of Synthon objects and number of rule
        """


__all__ = ['SynthonABC']
