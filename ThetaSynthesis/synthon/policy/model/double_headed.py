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
from torch.nn.functional import mse_loss
from torch.optim import Adam
from . import SorterNet


class DoubleHeadedNet(SorterNet):
    def __init__(self):
        super().__init__()

        self.policy_net = SorterNet.load_from_checkpoint(resource_stream(__name__, 'data/sorter.ckpt'))

        self.body = self.policy_net.body

        self.policy_head = Softmax(dim=0)

        self.value_head = Sequential(
            Linear(3956, 1),
        )

    def forward(self, x):
        assert isinstance(x, Tensor) and x.shape[0] == 4097

        fp, depth = x[:-1], x[-1]

        policy = self.body(fp)
        stack = hstack((policy, depth))

        value = self.value_head(stack)
        return policy, value

    def training_step(self, batch, batch_idx):
        loss = self._losses(batch, batch_idx)
        opt = self.optimizers()

        self.manual_backward(loss, opt)

        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._losses(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._losses(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=3e-4)
        return optimizer

    def _losses(self, batch, batch_idx):
        x, y = batch

        finger, depth = x[:, :-1], x[:, -1].reshape(-1, 1)
        stack = hstack((self.body(finger), depth))

        pred_value = self.value_head(stack)
        loss_value = mse_loss(pred_value, y)

        return loss_value


__all__ = ['DoubleHeadedNet']
