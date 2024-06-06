from torch.utils.data import Dataset
from torchvision.datasets import cifar

from .base import BasicDataset


class CIFAR10(BasicDataset):
    name = "cifar10"
    num_classes = 10
    shape = (3, 32, 32)

    mean = (0.4914, 0.4822, 0.4465)
    std_dev = (0.2023, 0.1994, 0.2010)
    num_train_samples = 50000
    num_val_samples = 10000

    def get_dataset(self, download: bool = True) -> Dataset:
        return cifar.CIFAR10(
            root=self.root_directory,
            train=self.is_train,
            transform=self.get_transform(),
            download=download,
        )
