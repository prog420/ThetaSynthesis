from abc import abstractmethod
from collections import deque
from functools import cached_property
from pickle import load
from typing import Tuple, List, Generator, TYPE_CHECKING

from CGRtools import CGRReactor
from MorganFingerprint import MorganFingerprint
from torch import from_numpy
from torch.nn import BCELoss

from .abc import SynthonABC
from .source import not_available, SimpleNet, TwoHeadedNet

if TYPE_CHECKING:
    from numpy import array

morgan = MorganFingerprint(length=4096, min_radius=2, max_radius=4, number_bit_pairs=4)

net = TwoHeadedNet(module=SimpleNet,
                   criterion=BCELoss,
                   device='cpu',
                   module__int_size=4096,
                   module__out_size=2272,
                   module__hid_size=(2000,),)
net.initialize()
net.load_params(f_params='ThetaSynthesis/source/params/twohead_params.pkl')


with open('source files/rules_reverse.pickle', 'rb') as f:
    rules = load(f)
reactors = [CGRReactor(rule, delete_atoms=True) for rule in rules]


class Synthon(SynthonABC):
    def descriptor(self):
        return from_numpy(morgan.transform([self.molecule])).float()

    @property
    def molecule(self):
        return self._molecule

    @abstractmethod
    def value(self) -> float:
        ...

    @cached_property
    def premolecules(self):
        prediction = self.__neural_network()
        sorted_indices = prediction.argsort()[::-1]
        for reactor in (reactors[x] for x in sorted_indices[:20]):
            for mol in reactor(self.molecule):
                products = mol.split()
                yield (type(self)(mol) for mol in products)

    @cached_property
    def probabilities(self) -> Tuple[float, ...]:
        return self.__neural_network()

    def __neural_network(self) -> "array":
        return net.predict(self.descriptor().unsqueeze(0)).squeeze()


class CombineSynthon(Synthon):
    # TODO: fix value's method
    @cached_property
    def value(self) -> float:
        return self._neural_network[1].item()


class StupidSynthon(Synthon):
    @property
    def value(self):
        return 1


class SlowSynthon(StupidSynthon):
    # TODO: refactor this
    @cached_property
    def value(self):
        """
        value get from rollout function
        """
        len_rollout = 10
        queue = deque(self.molecule)
        for _ in range(len_rollout):
            reactant = queue.popleft()
            descriptor = from_numpy(morgan.transform([reactant]))
            y = net.predict(descriptor)
            list_rules = [x for x in
                          sorted(enumerate([i.item() for i in y[0]]), key=lambda x: x[1], reverse=True)
                          ]
            list_rules = [(rules[x], y) for x, y in list_rules[:100]]
            for rule in list_rules:
                reactor = CGRReactor(rule[0], delete_atoms=True)
                products = list(reactor(reactant))
                if products:
                    queue.extend(not_available(products))
                    break
            if not queue:
                return 1
        return 0


__all__ = ['Synthon', 'CombineSynthon', 'SlowSynthon', 'StupidSynthon']
