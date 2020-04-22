from abc import ABC, abstractmethod
from CGRtools.containers import MoleculeContainer
from typing import List


class SynthonABC(MoleculeContainer, ABC):
    """
    class name maybe is not ideal option especially for chemists
    """
    @property
    @abstractmethod
    def descriptor(self) -> torch.FloatTensor:
        """
        make descriptor from MoleculeContainer and return it, nothing more
        """

    @abstractmethod
    def predict(self) -> (List, float):
        """
        the method take molecule's descriptor and get into neural network, return list of tuples,
        if tuples - query with probability, and evaluation score, or value
        """


class CombineSynthonABC(SynthonABC):
    """
    class for synthon with twohead combine neural network
    """


class SlowSynthonABC(SynthonABC):
    """
    synthon with onehead neural network, we get value from rollout
    """


class StupidSynthonABC(SlowSynthonABC):
    """
    synthon also with onehead neural network, inherit slow synthon, but value always the same number (1)
    """
