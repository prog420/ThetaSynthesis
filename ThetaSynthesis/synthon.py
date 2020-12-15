from abc import abstractmethod
from CGRtools import CGRReactor
from functools import cached_property
from MorganFingerprint import MorganFingerprint
from pickle import load
from torch import FloatTensor, load as torchload, device
from typing import Tuple, List
from .abc import SynthonABC
from .source import Chem, not_available, SimpleModel, SimpleNet, TwoHeadedNet


twohead_model = Chem(2006, 2273)
twohead_model.load_state_dict(torchload('source files/twohead_state_dict.pth'))
twohead_model.eval()

onehead_model = torchload('./source files/full_model.pth')
onehead_model.eval()

new_onehead_model = SimpleModel(inp_num=2048, hid_num=6000, out_num=2272)
new_onehead_model.load_state_dict(torchload('./source files/new_simple_model.pth', map_location=device('cpu')))
new_onehead_model.eval()

morgan = MorganFingerprint(length=2048, number_bit_pairs=4)

with open('source files/fitted_fragmentor.pickle', 'rb') as f:
    fragmentor = load(f)
with open('source files/rules_reverse.pickle', 'rb') as f:
    rules = load(f)
reactors = [CGRReactor(rule, delete_atoms=True) for rule in rules]


class Synthon(SynthonABC):
    @property
    def molecule(self):
        return self._molecule

    @property
    def value(self) -> float:
        ...

    @cached_property
    def premolecules(self) -> Tuple[Tuple['Synthon', ...], ...]:
        return tuple([x[0] for x in self._prods_probs()])

    @cached_property
    def probabilities(self) -> Tuple[float, ...]:
        return tuple([x[1] for x in self._prods_probs()])

    @cached_property
    def _descriptor(self):
        return FloatTensor(morgan.transform([self.molecule]).values)

    @abstractmethod
    def _probs(self):
        """
        raw vector of probabilities from neural network
        """

    def _prods_probs(self):
        """
        vector of pairs with Synthon object and probability of that Synthon
        the same for each child class
        take only the best 100 rules
        """
        # FIXME: rules fix!
        vector = sorted(list(enumerate(self._probs)), key=lambda x: x[1], reverse=True)
        # pairs with rule on reactor and probability
        best_100_rules_with_probs = [(reactors[x[0]], x[1]) for x in vector[:100]]
        out = []
        for pair in best_100_rules_with_probs:
            reactor, prob = pair
            list_products = list(reactor(self.molecule))
            if list_products:
                products = []
                for x in list_products:
                    products.extend(x.split())
                # create a new objects of the same class and append to out
                out.append([[type(self)(mol) for mol in products], prob])
        return out


class CombineSynthon(Synthon):
    @cached_property
    def value(self) -> float:
        return self._neural_network[1].item()

    @cached_property
    def _neural_network(self):
        return twohead_model(self._descriptor)

    @cached_property
    def _probs(self):
        return tuple(x.item() for x in self._neural_network[0][0])


class StupidSynthon(Synthon):
    # fixme: empty sequence
    @property
    def value(self):
        return 1

    @cached_property
    def _neural_network(self):
        return onehead_model(self._descriptor)

    @cached_property
    def _probs(self):
        return tuple([x.item() for x in self._neural_network[0]])


class SlowSynthon(StupidSynthon):
    @cached_property
    def value(self):
        """
        value get from rollout function
        """
        len_rollout = 10
        queue = [self.molecule]
        for _ in range(len_rollout):
            reactant = queue.pop(0)
            descriptor = FloatTensor(morgan.transform([reactant]))
            y = onehead_model(descriptor)
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
