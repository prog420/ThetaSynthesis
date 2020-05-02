from typing import Tuple

from .abc import SynthonABC


class Synthon(SynthonABC):
    @property
    def value(self) -> float:
        pass

    @property
    def probabilities(self) -> Tuple[float, ...]:
        pass

    @property
    def premolecules(self) -> Tuple[Tuple['SynthonABC', ...], ...]:
        pass


class CombineSynthon(Synthon):
    ...


class SlowSynthon(Synthon):
    ...


class StupidSynthon(Synthon):
    ...
