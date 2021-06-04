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
from torch.nn import ReLU, Sigmoid, Linear, Sequential, Softmax
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from torch.optim import Adam
from torchmetrics import MetricCollection, Accuracy


class FilterNet(LightningModule):
    def __init__(self):
        super().__init__()
        l1 = Linear(4096, 2000)
        l2 = Linear(2000, 3955)
        act = ReLU(inplace=True)

        self.metric = Accuracy()

        self.body = Sequential(l1, act, l2)
        self.policy_head = Sigmoid()

    def forward(self, x):
        return self.policy_head(self.body(x))

    def _predict(self, x):
        boo = self.forward(x) > 0.1
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
        optimizer = Adam(self.parameters(), lr=3e-4)
        return optimizer

    def _loss(self, batch, batch_idx):
        x, y = batch
        body = self.body(x)
        y_pred = self.policy_head(body)
        loss = binary_cross_entropy_with_logits(body, y)
        ba = self.metric(y_pred, y)
        return loss, ba


class SorterNet(LightningModule):
    def __init__(self):
        super().__init__()
        l1 = Linear(4096, 8000)
        l2 = Linear(8000, 3955)

        self.act = ReLU(inplace=True)
        self.head = Softmax()

        metrics = MetricCollection({
            str(n): Accuracy(top_k=n)
            for n
            in [1, 5, 10, 25, 50, 100]
        })

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.body = Sequential(l1, self.act, l2)

    def forward(self, x):
        return self.head(self.body(x))

    def training_step(self, batch, batch_idx):
        loss, m = self._loss_metrics(batch, self.train_metrics)
        self.log('train_loss', loss)
        self.log_dict(m, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, m = self._loss_metrics(batch, self.valid_metrics)
        self.log('valid_loss', loss)
        self.log_dict(m, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, m = self._loss_metrics(batch, self.test_metrics)
        self.log('test_loss', loss)
        self.log_dict(m, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _loss_metrics(self, batch, metrics: MetricCollection):
        x, y = batch
        y_pred = self.body(x)
        loss = cross_entropy(y_pred, y)
        y_pred = self.head(y_pred)
        m = metrics(y_pred, y)
        return loss, m


__all__ = ['FilterNet', 'SorterNet']
