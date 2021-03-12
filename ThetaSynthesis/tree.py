from CGRtools import MoleculeContainer
from typing import Type
from .abc import RetroTreeABC, SynthonABC
from .scroll import Scroll


class RetroTree(RetroTreeABC):
    def __init__(self, target: MoleculeContainer, /, synthon_class: Type[SynthonABC], depth=10, size=10000):
        synthon = synthon_class(target)
        super().__init__(Scroll((synthon,), {synthon}), depth=depth, size=size)

    def _add(self, pred: int, node: Scroll):
        """
        Add new node to tree.
        """
        succ = self._free_node
        self._nodes[succ] = node
        self._pred[succ] = pred
        self._succ[pred].add(succ)
        self._succ[succ] = set()
        self._free_node += 1

    def __next__(self):
        pass


__all__ = ['RetroTree']
