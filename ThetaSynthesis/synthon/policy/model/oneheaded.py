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
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import confusion_matrix
from torch.nn import ReLU, Sigmoid, Linear, Sequential
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam


class JustPolicyNet(LightningModule):
    def __init__(self):
        super().__init__()
        l1 = Linear(4096, 2000)
        l2 = Linear(2000, 2272)

        act = ReLU(inplace=True)

        self.body = Sequential(l1, act, l2)
        self.policy_head = Sigmoid()

    def forward(self, x):
        return self.policy_head(self.body(x))

    def _predict(self, x):
        boo = self.forward(x) > 0.5
        return boo.float()

    def training_step(self, batch, batch_idx):
        loss, ba = self._loss(batch, batch_idx)
        self.log('loss', loss)
        self.log('ba', ba, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ba = self._loss(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_ba', ba, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, ba = self._loss(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_ba', ba)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _loss(self, batch, batch_idx):
        x, y = batch
        y_pred = self.policy_head(self.body(x))
        loss = binary_cross_entropy(y_pred, y)

        y_pred_ = (y_pred > 0.5).float()
        (tn, fn), (fp, tp) = confusion_matrix(y_pred_, y, num_classes=2)
        ba = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
        return loss, ba


__all__ = ['JustPolicyNet']
