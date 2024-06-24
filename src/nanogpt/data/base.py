import os
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Specify paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))


class Dataset:
    @property
    @abstractmethod
    def batches_per_epoch(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def batch_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        raise NotImplementedError

    def __iter__(self):  # pragma: no cover
        return self

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self) -> tuple["torch.Tensor", "torch.Tensor"]:
        raise NotImplementedError
