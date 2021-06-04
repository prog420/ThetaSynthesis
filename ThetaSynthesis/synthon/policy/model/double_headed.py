# -*- coding: utf-8 -*-
#
#  Copyright 2020-2021 Alexander Sizov <murkyrussian@gmail.com>
#  Copyright 2021 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of ThetaSynthesis.
#
#  ThetaSynthesis is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from pkg_resources import resource_stream
from torch import hstack, Tensor
from torch.nn import Linear, Sequential, Softmax
from torch.nn.functional import kl_div, mse_loss
from torch.optim import Adam
from . import SorterNet


class DoubleHeadedNet(SorterNet):
    def __init__(self):
        super().__init__()

        self.policy_net = SorterNet.load_from_checkpoint(resource_stream(__name__, 'data/sorter.ckpt'))

        self.body = self.policy_net.body

        self.policy_head = Softmax(dim=0)

        self.value_head = Sequential(
            Linear(2273, 1),
        )

    def forward(self, x):
        if isinstance(x, Tensor) and x.shape[0] == 4096:
            return self.policy_head(self.body(x))
        elif isinstance(x, Tensor) and x.shape[0] == 4097:
            x_policy, x_value = x[:-1], x[-1]
        elif isinstance(x, tuple):
            x_policy, x_value = x
        else:
            raise TypeError

        policy = self.policy_net(x_policy)
        stack = hstack((policy, x_value))

        value = self.value_head(stack)
        return policy, value

    def _predict(self, x):
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        loss_policy, loss_value = self._losses(batch, batch_idx)
        opt = self.optimizers()

        self.manual_backward(loss_policy, opt, retain_graph=True)
        self.manual_backward(loss_value, opt)

        self.log('loss_policy', loss_policy)
        self.log('loss_value', loss_value)

        loss = loss_policy + loss_value
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_policy, loss_value = self._losses(batch, batch_idx)

        self.log('val_loss_policy', loss_policy)
        self.log('val_loss_value', loss_value)

        loss = loss_policy + loss_value
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss_policy, loss_value = self._losses(batch, batch_idx)

        self.log('test_loss_policy', loss_policy)
        self.log('test_loss_value', loss_value)

        loss = loss_policy + loss_value
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=3e-4)
        return optimizer

    def _losses(self, batch, batch_idx):
        x, y = batch

        finger, depth = x[:, :-1], x[:, -1].reshape(-1, 1)
        y_policy, y_value = y[:, :-1], y[:, -1].reshape(-1, 1)

        pred_policy = self.policy_net(finger)
        stack = hstack((self.body(finger), depth))

        pred_value = self.value_head(stack)

        loss_policy = kl_div(pred_policy, y_policy, reduction='batchmean')
        loss_value = mse_loss(pred_value, y_value)

        return loss_policy, loss_value

    def _positive_exps_learn(self):
        ...

    def _negative_exps_learn(self):
        ...


__all__ = ['DoubleHeadedNet']
