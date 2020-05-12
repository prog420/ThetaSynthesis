from CGRtools.containers import MoleculeContainer
from functools import cached_property
from model import Chem
from pickle import load
from torch import FloatTensor
from typing import Tuple
from .abc import SynthonABC


class Synthon(SynthonABC):
    __slots__ = ('_fragmentor', '_model', '__dict__')

    def __init__(self, molecule):
        if not isinstance(molecule, MoleculeContainer):
            raise TypeError('Synthon is only MoleculeContainer')
        self.__new__(self, molecule)
        self._model = Model()
        with open('./source files/fitted_fragmentor.pickle', 'rb') as f:
            self._fragmentor = load(f)

    @cached_property
    def value(self) -> float:
        return self._neural_network()[1].item()

    @cached_property
    def probabilities(self) -> Tuple[float, ...]:
        return tuple([x.item() for x in self._neural_network()[0][0]])

    @cached_property
    def premolecules(self) -> Tuple[Tuple['SynthonABC', ...], ...]:
        vector = self.probabilities
        return

    @cached_property
    def _get_descriptor(self):
        return FloatTensor(self._fragmentor.transform([self._molecule]).values)

    def _neural_network(self):
        return self._model(self._get_descriptor)


class CombineSynthon(Synthon):
    ...


class StupidSynthon(Synthon):
    ...


class SlowSynthon(StupidSynthon):
    ...
