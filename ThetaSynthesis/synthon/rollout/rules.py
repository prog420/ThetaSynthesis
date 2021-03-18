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
from pickle import load
from typing import TYPE_CHECKING, Tuple, List

from CGRtools import Reactor
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from StructureFingerprint import LinearFingerprint
from torch import from_numpy, sort
from torch.nn import ReLU, Sigmoid, Linear, Sequential
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam

if TYPE_CHECKING:
    from CGRtools import MoleculeContainer


class RulesNet(LightningModule):
    def __init__(self):
        super().__init__()
        l1 = Linear(4096, 2000)
        l2 = Linear(2000, 2272)

        act = ReLU(inplace=True)

        self.frag = LinearFingerprint(length=4096, min_radius=2, max_radius=4, number_bit_pairs=4)
        self.reactors = [Reactor(x, delete_atoms=True)
                         for x in load(resource_stream(__name__, 'data/rules_reverse.pickle'))]

        self.body = Sequential(l1, act, l2)
        self.policy_head = Sigmoid()

    def forward(self, x):
        return self.policy_head(self.body(self.transform(x)))

    def predict(self, x):
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
        ba = accuracy(y_pred_, y, class_reduction='macro')
        return loss, ba

    def transform(self, x: "MoleculeContainer"):
        return from_numpy(self.frag.transform([x])).float()

    def get_reactors(self, x: 'MoleculeContainer') -> List[Tuple[float, Reactor]]:
        # ordered by patent's frequencies reactors
        rules_bit_vector = self.forward(x)
        values, indices = sort(rules_bit_vector, descending=True)
        return [(p, self.reactors[index.item()]) for prob, index
                in zip(values.squeeze(0), indices.squeeze(0)) if (p := prob.item()) > .1][:50]


__all__ = ['RulesNet']
