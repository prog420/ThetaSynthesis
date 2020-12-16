from CGRtools.containers import MoleculeContainer, ReactionContainer
from functools import cached_property
from typing import Tuple
from .abc import ScrollABC
from .source import not_available


class Scroll(ScrollABC):
    @cached_property
    def premolecules(self) -> Tuple['Scroll', ...]:
        """
        return new scrolls from that scroll
        """
        in_scroll = list(self._synthons)
        target = in_scroll.pop(0)
        new_depth = self._depth + 1
        for gen, prob in zip(target.premolecules(), target.probabilities()):
            mols = tuple(gen)
            reaction = ReactionContainer(tuple(x.molecule for x in mols), [target.molecule])
            child_scroll = Scroll(synthons=tuple(in_scroll + list(self._filter(mols))),
                                  reaction=reaction,
                                  probability=prob,
                                  depth=new_depth)
            yield child_scroll

    @property
    def worse_value(self):
        return min([x.value for x in self._synthons])

    def _filter(self, synthons):
        """
        return only commercially unavailable for molecules in input synthons
        """
        comm_molecules = not_available((x.molecule for x in synthons))
        return (x for x in synthons if x.molecule in comm_molecules)


__all__ = ['Scroll']
