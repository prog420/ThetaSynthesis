from typing import Iterable, TYPE_CHECKING, Any, Union, List, Optional

import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .tree import RetroTree

if TYPE_CHECKING:
    from CGRtools.containers import MoleculeContainer


class RetroTreeDataModule(LightningDataModule):
    # TODO return positive and negative examples
    # TODO add partial_fit of tree
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

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

    @staticmethod
    def __collect_from_tree(x: Iterable[MoleculeContainer, ]):
        positives = []
        negatives = []
        for mol in x:
            tree = RetroTree(mol)
            positives.extend(set(sum(tree, [])))
            negatives.extend(tree.generate_negative())
        return positives, negatives


__all__ = ['RetroTreeDataModule']
