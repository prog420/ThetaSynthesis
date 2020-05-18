from CGRtools.containers import MoleculeContainer, ReactionContainer
from typing import Tuple
from .synthon import Synthon
from .abc import ScrollABC


class Scroll(ScrollABC):
    @property
    def premolecules(self) -> Tuple['ScrollABC', ...]:
        in_scroll = list(self._synthons)
        target = in_scroll.pop(0)
        new_synthons = target.premolecules
        new_depth = self._depth + 1
        scrolls = []
        for synthons in new_synthons:
            reaction = ReactionContainer(self._extract(synthons), [target.get_molecule])
            child_scroll = Scroll(in_scroll + list(synthons), reaction, new_depth)
            scrolls.append(child_scroll)
        return tuple(scrolls)

    @property
    def worse_value(self):
        return min([x.value for x in self._synthons])

    def _extract(self, synthons:Tuple[Synthon, ...]) -> Tuple[MoleculeContainer, ...]:
        return tuple([x.get_molecule for x in synthons])


__all__ = ['Scroll']
