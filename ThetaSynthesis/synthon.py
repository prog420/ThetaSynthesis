from CGRtools import CGRReactor
from functools import cached_property
from pickle import load
from torch import FloatTensor, load as torchload
from typing import Tuple, List
from .abc import SynthonABC
from .source import Chem


twohead_model = Chem(2006, 2273)
twohead_model.load_state_dict(torchload('source files/twohead_state_dict.pth'))
twohead_model.eval()

with open('source files/fitted_fragmentor.pickle', 'rb') as f:
    fragmentor = load(f)
with open('source files/rules_reverse.pickle', 'rb') as f:
    rules = load(f)


class Synthon(SynthonABC):
    @property
    def get_molecule(self):
        return self._molecule

    @property
    def value(self) -> float:
        pass

    @property
    def probabilities(self) -> Tuple[float, ...]:
        pass

    @property
    def premolecules(self) -> Tuple[Tuple['Synthon', ...], ...]:
        pass

    @cached_property
    def _get_descriptor(self):
        return FloatTensor(fragmentor.transform([self._molecule]).values)


class CombineSynthon(Synthon):
    @cached_property
    def premolecules(self) -> Tuple[Tuple['Synthon', ...], ...]:
        return tuple([x[0] for x in self._prods_probs])

    @cached_property
    def probabilities(self) -> Tuple[float, ...]:
        return tuple([x[1] for x in self._prods_probs])

    @cached_property
    def value(self) -> float:
        return self._neural_network[1].item()

    @cached_property
    def _neural_network(self):
        return twohead_model(self._get_descriptor)

    @property
    def _probs(self):
        return tuple([x.item() for x in self._neural_network[0][0]])

    @property
    def _prods_probs(self):
        vector = sorted(list(enumerate(self._probs)), key=lambda x: x[1], reverse=True)
        best_100_rules_with_probs = [(rules[x[0]], x[1]) for x in vector[:100]]
        out = []
        for pair in best_100_rules_with_probs:
            rule, prob = pair
            reactor = CGRReactor(rule, delete_atoms=True)
            list_products = list(reactor(self._molecule))
            if list_products:
                products = []
                for x in list_products:
                    products.append(x.split())
                out.append([[CombineSynthon(mol[0]) for mol in products], prob])
        return out


class StupidSynthon(Synthon):
    @cached_property
    def probabilities(self) -> Tuple[float, ...]:
        return tuple([x.item() for x in self._neural_network()[0][0]])

    @property
    def value(self):
        return 1

    def _neural_network(self):
        return


class SlowSynthon(StupidSynthon):
    ...


__all__ = ['Synthon', 'CombineSynthon', 'SlowSynthon', 'StupidSynthon']
