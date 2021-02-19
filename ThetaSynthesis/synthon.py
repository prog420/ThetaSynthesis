from abc import abstractmethod
from collections import deque
from functools import cached_property
from pickle import load
from typing import TYPE_CHECKING

from CGRtools import CGRReactor
from StructureFingerprint import LinearFingerprint
from torch import from_numpy, sort

from .abc import SynthonABC
from .source import not_available, JustPolicyNet

if TYPE_CHECKING:
    from torch import Tensor
    from CGRtools.containers import MoleculeContainer

morgan = LinearFingerprint(length=4096, min_radius=2, max_radius=4, number_bit_pairs=4)

net = JustPolicyNet.load_from_checkpoint('ThetaSynthesis/source/net.ckpt')
net.eval()

with open('source files/rules_reverse.pickle', 'rb') as f:
    rules = load(f)
reactors = [CGRReactor(rule, delete_atoms=True) for rule in rules]


class Synthon(SynthonABC):
    @property
    def molecule(self) -> "MoleculeContainer":
        return self._molecule

    def premolecules(self, top_n: int = 10):
        return tuple(tuple(type(self)(mol) for mol in reactor(self.molecule) for mol in mol.split())
                     for reactor in (reactors[idx] for idx in self.__sorted[1][:top_n]))

    def probabilities(self, top_n: int = 10):
        return tuple(prob for prob in self.__sorted[0][:top_n])

    @abstractmethod
    def value(self, **kwargs):
        ...

    def _descriptor(self) -> "Tensor":
        return from_numpy(morgan.transform([self.molecule])).float()

    def _predict(self):
        return net.predict(self._descriptor())

    @cached_property
    def __sorted(self):
        sorted_, values = sort(self._predict(), descending=True)
        return sorted_.squeeze(), values.squeeze()


class CombineSynthon(Synthon):
    def value(self, **kwargs):
        return super()._predict()[1].item()


class StupidSynthon(Synthon):
    def value(self, **kwargs):
        return 1.


class SlowSynthon(StupidSynthon):
    def value(self, **kwargs):
        """
        value get from rollout function
        """
        queue = deque([self])
        for _ in range(kwargs['roll_len'] - kwargs['depth']):
            reactant = queue.popleft()
            queue.extend(i for x in not_available(reactant.premolecules(1)) for i in x)
            if not queue:
                return 1.
        return 0.


__all__ = ['Synthon', 'CombineSynthon', 'SlowSynthon', 'StupidSynthon']
