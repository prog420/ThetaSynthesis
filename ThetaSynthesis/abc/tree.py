from abc import ABC, abstractmethod


class RetroTreeABC(ABC):
    def __init__(self):
        ...

    def select(self):
        ...

    def expand(self):
        ...
