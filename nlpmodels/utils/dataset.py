"""
This module contains the abstract base class for our datasets.
"""

# pylint: disable=missing-docstring

from abc import abstractmethod, ABC

from torch.utils.data import Dataset


class AbstractNLPDataset(Dataset, ABC):
    """
        Abstract base class for dataset class.
    """

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @classmethod
    @abstractmethod
    def get_training_data(cls, *args):
        pass

    @classmethod
    @abstractmethod
    def get_testing_data(cls, *args):
        pass

    @classmethod
    @abstractmethod
    def get_training_dataloader(cls, *args):
        pass

    @classmethod
    @abstractmethod
    def get_testing_dataloader(cls, *args):
        pass
