"""
This submodule contains data preparation code for some of the datasets used with our models,
i.e. MNIST, CIFAR 10 and 100 and ImageNet.
"""

from typing import List, Type

from .base import BasicDataset
from .mnist import MNIST
from .cifar10 import CIFAR10
from .svhn import SVHN

__all__ = [
    "BasicDataset",
    "dataset_from_name",
    "dataset_names",
    "MNIST",
    "CIFAR10",
    "SVHN"
]

_datasets = {
    MNIST.name: MNIST,
    CIFAR10.name: CIFAR10,
    SVHN.name: SVHN
}


def dataset_from_name(name: str) -> Type[BasicDataset]:
    """returns the dataset to which the name belongs to (name has to be the value of the datasets
    name-attribute)

    Args:
        name (str): name of the dataset

    Raises:
        ValueError: raised if no dataset under that name was found

    Returns:
        dataset: the dataset
    """
    if name in _datasets:
        return _datasets[name]

    raise Exception(f"unknown dataset: {name}")


def dataset_names() -> List[str]:
    """getter for list of dataset names for argparse

    Returns:
        List: the dataset names
    """
    return list(_datasets.keys())
