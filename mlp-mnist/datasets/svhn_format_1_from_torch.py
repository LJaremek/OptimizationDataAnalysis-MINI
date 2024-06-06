from torch.utils.data import Dataset
from torchvision import datasets

from .base import BasicDataset


class SVHN(BasicDataset):
    name = "svhn"
    num_classes = 10

    shape = (3, 32, 32)

    mean = (0.4377, 0.4438, 0.4728)
    std_dev = (0.4377, 0.4438, 0.4728)
    num_train_samples = 50000
    num_val_samples = 10000


    def get_dataset(self, download: bool = True) -> Dataset:
        split = "test"
        if self.is_train == True:
            split = "test"
        return datasets.SVHN(
            root=self.root_directory,
            split=split,
            transform=self.get_transform(),
            download=download,
        )
