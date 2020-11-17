import pytorch_lightning as pl
import torch

from discriminator import Discriminator


class NClassesClassifier(pl.LightningModule):
    def __init__(self, discriminator: Discriminator):
        super().__init__()

        self.disc = discriminator
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.disc(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, labels = batch

        prediction = self.disc(real)
        loss = self.criterion(prediction, labels)

        self.log_dict({"loss": loss})

        return loss
