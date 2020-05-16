from typing import Tuple

from .synthon import Synthon
from .abc import ScrollABC


class Scroll(ScrollABC):
    @property
    def premolecules(self) -> Tuple['ScrollABC', ...]:
        return

    @property
    def worse_value(self):
        return min([x.value for x in self._synthons])


__all__ = ['Scroll']
