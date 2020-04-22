from .abc import SynthonABC


class Synthon(SynthonABC):
    @property
    def descriptor(self):
        return

    def predict(self):
        pass


class CombineSynthon(Synthon):
    ...


class SlowSynthon(Synthon):
    ...


class StupidSynthon(Synthon):
    ...
