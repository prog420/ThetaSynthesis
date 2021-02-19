from CGRtools.containers import ReactionContainer
from functools import cached_property
from typing import Tuple
from .abc import ScrollABC
from .source import not_available


class Scroll(ScrollABC):
    def premolecules(self) -> Tuple['Scroll', ...]:
        """
        return new scrolls from that scroll
        """
        in_scroll = list(self._synthons)
        target = in_scroll.pop(0)
        new_depth = self._depth + 1
        scrolls = []
        for tpl, prob in zip(target.premolecules(), target.probabilities()):
            curr_target = target.molecule
            reaction = ReactionContainer(tuple(x.molecule for x in tpl), [curr_target])
            child_scroll = Scroll(synthons=tuple(in_scroll + list(self._filter(tpl))),
                                  reaction=reaction,
                                  probability=prob,
                                  depth=new_depth,
                                  parents=self._parents | {curr_target})
            scrolls.append(child_scroll)
        return tuple(scrolls)

    @cached_property
    def worse_value(self):
        return min(x.value(roll_len=10, depth=self.depth) for x in self._synthons) if self._synthons else 1.

    def _filter(self, synthons):
        """
        return only commercially unavailable for molecules in input synthons
        """
        return tuple(x for x in synthons if x.molecule in not_available(x.molecule for x in synthons))


__all__ = ['Scroll']
