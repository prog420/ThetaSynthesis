from abc import abstractmethod
from collections import deque
from functools import cached_property
from pickle import load
from typing import Tuple, List, Generator, TYPE_CHECKING

from CGRtools import CGRReactor
from MorganFingerprint import MorganFingerprint
from torch import from_numpy, sort
from torch.nn import BCELoss

from .abc import SynthonABC
from .source import not_available, SimpleNet, TwoHeadedNet

if TYPE_CHECKING:
    from numpy import array
    from CGRtools.containers import MoleculeContainer

morgan = MorganFingerprint(length=4096, min_radius=2, max_radius=4, number_bit_pairs=4)

net = TwoHeadedNet(module=SimpleNet,
                   criterion=BCELoss,
                   device='cpu',
                   module__int_size=4096,
                   module__out_size=2272,
                   module__hid_size=(2000,), )
net.initialize()
net.load_params(f_params='ThetaSynthesis/source/params/twohead_params.pkl')

with open('source files/rules_reverse.pickle', 'rb') as f:
    rules = load(f)
reactors = [CGRReactor(rule, delete_atoms=True) for rule in rules]


class Synthon(SynthonABC):
    @property
    def molecule(self) -> "MoleculeContainer":
        return self._molecule

    def premolecules(self, top_n: int = 10):
        return tuple(tuple(type(self)(mol) for mol in reactor(self.molecule) for mol in mol.split())
                     for reactor in (reactors[idx] for idx in self.__sorted_pairs[1][:top_n]))

    def probabilities(self, top_n: int = 10):
        return tuple(prob for prob in self.__sorted_pairs[0][:top_n])

    @abstractmethod
    def value(self, **kwargs):
        ...

    def descriptor(self) -> "array":
        return from_numpy(morgan.transform([self.molecule])).float()

    @cached_property
    def __sorted_pairs(self):
        probs = self._neural_network()[0].squeeze(0).squeeze(0)
        values, indices = sort(probs, descending=True)
        return values, indices

    def _neural_network(self):
        return net.forward(self.descriptor().unsqueeze(0))


class CombineSynthon(Synthon):
    def value(self, **kwargs):
        return super()._neural_network()[1].squeeze(0).squeeze(0).item()


class StupidSynthon(Synthon):
    def value(self, **kwargs):
        return 1.


class SlowSynthon(StupidSynthon):
    def value(self, **kwargs):
        """
        value get from rollout function
        :param **kwargs:
        """
        queue = deque([self])
        for _ in range(kwargs['roll_len'] - kwargs['depth']):
            reactant = queue.popleft()
            queue.extend(i for x in not_available(reactant.premolecules(1)) for i in x)
            if not queue:
                return 1.
        return 0.


__all__ = ['Synthon', 'CombineSynthon', 'SlowSynthon', 'StupidSynthon']
