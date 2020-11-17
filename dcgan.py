import pytorch_lightning as pl
import torch
import torchvision

from discriminator import Discriminator
from generator import Generator


class DCGAN(pl.LightningModule):
    def __init__(self, generator: Generator, discriminator: Discriminator):
        super().__init__()

        self.gen = generator
        self.disc = discriminator
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, noise):
        return self.gen(noise)

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.gen.parameters(), lr=3e-4, betas=(0.5, 0.999))
        opt_disc = torch.optim.Adam(self.disc.parameters(), lr=3e-4, betas=(0.5, 0.999))

        return [opt_gen, opt_disc], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        if optimizer_idx == 0:
            return self.train_gen(real)
        if optimizer_idx == 1:
            return self.train_disc(real)

    def train_gen(self, real):
        noise = self.gen.gen_noize(n_samples=len(real))
        fake = self.gen(noise)
        pred = self.disc(fake)

        gen_loss = self.criterion(pred, torch.ones_like(pred))
        self.log_dict({"gen_loss": gen_loss})

        return gen_loss

    def train_disc(self, real):
        noise = self.gen.gen_noize(n_samples=len(real))
        fake = self.gen(noise).detach()
        fake_pred = self.disc(fake)
        disc_gen_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))

        real_pred = self.disc(real)
        disc_real_loss = self.criterion(real_pred, torch.ones_like(real_pred))

        disc_loss = (disc_gen_loss + disc_real_loss) / 2
        self.log_dict({"disc_loss": disc_loss})

        return disc_loss

    def on_epoch_end(self) -> None:
        fake = self.gen(self.gen.gen_noize())

        img_grid = torchvision.utils.make_grid(fake)
        self.logger.experiment.add_image('generated_images', img_grid, self.current_epoch)
