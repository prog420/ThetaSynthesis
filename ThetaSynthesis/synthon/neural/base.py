from pickle import load
from typing import Iterator, Tuple, TYPE_CHECKING

from CGRtools import Reactor
from functools import cached_property
from StructureFingerprint import LinearFingerprint
from torch import from_numpy, sort

from ..abc import SynthonABC

if TYPE_CHECKING:
    from torch import Tensor

frag = LinearFingerprint(length=4096, min_radius=2, max_radius=4, number_bit_pairs=4)

with open('files/bb.pickle', 'rb') as f:
    stock = frozenset(load(f))

with open('files/rules_reverse.pickle', 'rb') as f:
    reactors = [Reactor(x, delete_atoms=True) for x in load(f)]


class NeuralSynthon(SynthonABC):
    __slots__ = ('_network', )

    def __init__(self, molecule):
        super().__init__(molecule)

    def __iter__(self) -> Iterator[Tuple[float, Tuple['SynthonABC', ...]]]:
        for prob, num in zip(self._sorted_predict):
            for react in reactors[num]([self.molecule]):
                mols = []
                for mol in react.products:
                    mol.kekule()
                    mol.thiele()
                    mols.append(mol)
                yield prob, tuple(type(self)(mol) for mol in mols)

    def __bool__(self):
        return self.molecule in stock

    def __float__(self):
        ...

    @cached_property
    def _fingerprint(self) -> "Tensor":
        return from_numpy(frag.transform([self.molecule])).float()

    @cached_property
    def _sorted_predict(self) -> Tuple["Tensor", "Tensor"]:
        return sort(self._network.predict(self._fingerprint), descending=True)
