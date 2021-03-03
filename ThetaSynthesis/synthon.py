from abc import abstractmethod
from collections import deque
from functools import cached_property
from pickle import load
from typing import Generator, Tuple, TYPE_CHECKING

from CGRtools import CGRReactor
from StructureFingerprint import LinearFingerprint
from torch import from_numpy, sort

from .abc import SynthonABC
from .source import not_available
from .model import JustPolicyNet

if TYPE_CHECKING:
    from torch import Tensor
    from CGRtools.containers import MoleculeContainer

fragmentor = LinearFingerprint(length=4096, min_radius=2, max_radius=4, number_bit_pairs=4)

net = JustPolicyNet.load_from_checkpoint('ThetaSynthesis/source/net.ckpt')
net.eval()

with open('ThetaSynthesis/source/rules_reverse.pickle', 'rb') as f:
    rules = load(f)
reactors = [CGRReactor(rule, delete_atoms=True) for rule in rules]


class Synthon(SynthonABC):
    @property
    def molecule(self) -> "MoleculeContainer":
        return self._molecule

    def __str__(self):
        return str(self.molecule)

    def __len__(self):
        return len(self.molecule)

    def premolecules(self, top_n: int = 100) -> Generator[Tuple[Tuple, float], None, None]:
        yield from (
            tuple(
                [[type(self)(each) for each in mol.split()], idx.item()]
            )
            for idx in self.__sorted[1][:top_n]
            for mol in reactors[(idx.item())](self.molecule)
        )

    def probabilities(self, top_n: int = 10) -> Generator[float, None, None]:
        yield from (prob.item() for prob in self.__sorted[0][:top_n])

    @abstractmethod
    def value(self, **kwargs):
        ...

    def descriptor(self) -> "Tensor":
        return from_numpy(fragmentor.transform([self.molecule])).float()

    def _predict(self):
        return net.predict(self.descriptor())

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
        for step in range(kwargs['roll_len'] - kwargs['depth']):
            reactant = queue.popleft()
            for synthons, idx in reactant.premolecules():
                if synthons:
                    queue.extend(x for x in not_available(synthons))
                    break
            if not queue:
                return 1. ** step
        return -1.


__all__ = ['Synthon', 'CombineSynthon', 'SlowSynthon', 'StupidSynthon']
