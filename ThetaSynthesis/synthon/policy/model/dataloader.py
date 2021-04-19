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
