from typing import Union, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets.celeba import CelebA
from torchvision.transforms import transforms

class CelebADataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=128):
        super().__init__()

        self.root = root
        self.batch_size = batch_size

        self.transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        self.train_dataset = CelebA(root="data", split="train", download=True, transform=self.transforms)
        self.valid_dataset = CelebA(root="data", split="valid", download=True, transform=self.transforms)
        self.test_dataset = CelebA(root="data", split="test", download=True, transform=self.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size)






