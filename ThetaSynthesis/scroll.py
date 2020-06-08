from CGRtools.containers import MoleculeContainer, ReactionContainer
from typing import Tuple
from .abc import ScrollABC


class Scroll(ScrollABC):
    @property
    def premolecules(self) -> Tuple['Scroll', ...]:
        in_scroll = list(self._synthons)
        target = in_scroll.pop(0)
        new_synthons = target.premolecules
        new_depth = self._depth + 1
        scrolls = []
        for synthons in new_synthons:
            reaction = ReactionContainer(self._extract(synthons), [target.get_molecule])
            child_scroll = Scroll(in_scroll + list(synthons), reaction, propability, new_depth)
            scrolls.append(child_scroll)
        return tuple(scrolls)

    @property
    def worse_value(self):
        return min([x.value for x in self._synthons])

    def _extract(self, synthons) -> Tuple[MoleculeContainer, ...]:
        return tuple([x.get_molecule for x in synthons])


__all__ = ['Scroll']
