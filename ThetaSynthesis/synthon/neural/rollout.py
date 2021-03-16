from .base import NeuralSynthon
from ..models import JustPolicyNet

net = JustPolicyNet.load_from_checkpoint('data/net.ckpt')
net.eval()


class RolloutSynthon(NeuralSynthon):
    def __init__(self, molecule):
        super().__init__(molecule)
        ...

    def __float__(self):
        ...
