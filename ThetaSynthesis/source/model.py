from skorch.net import NeuralNet
from torch.nn import Sequential, Linear, Sigmoid, ReLU, Module, Softmax, Tanh
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


class SimpleModel(Module):
    def __init__(self, inp_num, out_num, hid_num):
        super().__init__()
        self.l1 = Linear(inp_num, hid_num)
        kaiming_normal_(self.l1.weight)
        zeros_(self.l1.bias)
        self.activ = ReLU(inplace=True)
        self.l2 = Linear(hid_num, out_num)
        self.soft = Softmax()

    def forward(self, x):
        x = self.l1(x)
        x = self.activ(x)
        x = self.l2(x)

        return x


class SimpleNet(Module):
    def __init__(self, int_size, out_size, hid_size, activation=ReLU):
        super().__init__()
        sizes = (int_size, *hid_size, out_size)
        for n, (i, j) in enumerate(zip(sizes, sizes[1:]), start=1):
            l = Linear(i, j)
            kaiming_normal_(l.weight)
            zeros_(l.bias)
            setattr(self, f'l{n}', l)
        self.last = Linear(sizes[-1], 1)
        self.__size = len(sizes) - 1
        self.af = activation()
        self.sg = Sigmoid()
        self.th = Tanh()

    def forward(self, x):
        for n in range(1, self.__size):
            x = getattr(self, f'l{n}')(x)
            x = self.af(x)
        x = getattr(self, f'l{self.__size}')(x)
        return self.sg(x), self.th(self.last(x))


class TwoHeadedNet(NeuralNet):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        policy, value = y_pred
        loss_policy = super().get_loss(policy, y_true, *args, **kwargs)
        return loss_policy
