import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets.celeba import CelebA
from torchvision.transforms import transforms

class MNISTDataModule(pl.LightningDataModule):
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
        self.dataset = CelebA(root="data", download=True, transform=self.transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

