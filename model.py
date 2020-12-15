from torch.nn import Sequential, Linear, Sigmoid, ReLU, Module, Softmax
from torch.nn.init import kaiming_normal_, zeros_


class Chem(Module):
    def __init__(self, in_f, num_rules):
        super().__init__()
        self.in_f = in_f
        self.num_rules = num_rules
        self.model_body = Sequential(
            Linear(in_f, 4000),
            ReLU(inplace=True),
            Linear(4000, num_rules - 1)
        )
        self.rules_head = Sigmoid()

        self.value_head = Sequential(
            ReLU(),
            Linear(num_rules - 1, 1000),
            ReLU(inplace=True),
            Linear(1000, 1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.model_body(x)
        rules = self.rules_head(x)
        values = self.value_head(x)
        return rules, values
