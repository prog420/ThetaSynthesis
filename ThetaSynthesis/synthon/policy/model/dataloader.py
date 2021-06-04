# -*- coding: utf-8 -*-
#
#  Copyright 2021 Alexander Sizov <murkyrussian@gmail.com>
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
from itertools import chain, repeat
from typing import Any, Union, List, Optional, TYPE_CHECKING, Iterable

import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ....tree import RetroTree
from .. import DoubleHeadedSynthon

if TYPE_CHECKING:
    from CGRtools import MoleculeContainer


class RetroTreeDataModule(LightningDataModule):
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        pass

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def transfer_batch_to_device(self, batch: Any, device: Optional[torch.device] = None) -> Any:
        pass

    @staticmethod
    def __collect_from_tree(x: Iterable['MoleculeContainer']):
        for mol in x:
            tree = RetroTree(mol, synthon_class=DoubleHeadedSynthon)
            win_terminals = list(tree)
            winner_nodes = list(zip(chain.from_iterable([tree.chain_to_node(node) for node in win_terminals]), repeat(1.)))
            loser_nodes = zip(
                sorted(tree._nodes,
                       key=lambda k: tree._visits[k],
                       reverse=True)[:min(len(winner_nodes), 10)],
                repeat(-1.))
            for example in chain(winner_nodes, loser_nodes):
                node, value = example
                yield [tree._nodes[node].current_synthon._bit_string, ..., ...]


__all__ = ['RetroTreeDataModule']
