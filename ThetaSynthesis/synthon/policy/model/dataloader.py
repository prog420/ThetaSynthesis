from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader
from typing import Union, List, Optional, TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from torch import Tensor


class RetroTreeDataModule(LightningDataModule):
    def __init__(self, tensors: Iterable['Tensor'], batch_size: int = 1):
        super().__init__()
        self.tensors = tensors
        self._batch_size = batch_size

    def prepare_data(self, *args, **kwargs):
        self.X, self.y = self.tensors

    def setup(self, stage: Optional[str] = None):
        self.train_set = TensorDataset(self.X, self.y)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self._batch_size, num_workers=12)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        pass


__all__ = ['RetroTreeDataModule']
