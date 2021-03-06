import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

from discriminator import Discriminator
from generator import Generator


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def combine_vectors(x, y):
    return torch.cat((x, y), dim=1).float()

def get_dimensions(noize_dim, image_size, n_classes):
    generator_input_dim = noize_dim + n_classes
    discriminator_input_dim = image_size + n_classes
    return generator_input_dim, discriminator_input_dim


class ConditionalGAN(pl.LightningModule):
    def __init__(self, generator: Generator, discriminator: Discriminator, noize_dim=10):
        super().__init__()
        self.generator = generator.apply(weights_init)
        self.discriminator = discriminator.apply(weights_init)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.noize_dim = noize_dim

        self.last_fake = None

    def forward(self, noise):
        return self.generator(noise)

    def train_gen(self, real, one_hot_labels, image_one_hot_labels):

        noise = self.generator.gen_noize(len(real), noize_dim=self.noize_dim, device=self.device)
        noise_and_labels = combine_vectors(noise, one_hot_labels.float())
        fake = self.generator(noise_and_labels)
        fake_images_and_labels = combine_vectors(fake, image_one_hot_labels)
        fake_pred = self.discriminator(fake_images_and_labels)
        gen_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        self.log_dict({'gen_loss': gen_loss})

        return gen_loss

    def train_disc(self, real, one_hot_labels, image_one_hot_labels):
        noise = self.generator.gen_noize(len(real), noize_dim=self.noize_dim, device=self.device)
        noise_and_labels = combine_vectors(noise, one_hot_labels.float())
        fake = self.generator(noise_and_labels).detach()
        self.last_fake = fake

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        real_image_and_labels = combine_vectors(real, image_one_hot_labels)
        fake_pred = self.discriminator(fake_image_and_labels)
        real_pred = self.discriminator(real_image_and_labels)

        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))

        disc_loss = (fake_loss + real_loss) / 2
        self.log_dict({'disc_loss': disc_loss})

        return disc_loss

    def training_step(self, batch, batch_idx, optimizer_idx):

        real, labels = batch
        optimizer = self.optimizers()[optimizer_idx]

        one_hot_labels = labels
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, 64, 64)

        if optimizer_idx == 0:
            return self.train_gen(real, one_hot_labels, image_one_hot_labels)
        if optimizer_idx == 1:
            return self.train_disc(real, one_hot_labels, image_one_hot_labels)

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        return [opt_gen, opt_disc], []

    def on_epoch_end(self) -> None:
        img_grid = torchvision.utils.make_grid(self.last_fake)
        self.logger.experiment.add_image('generated_images', img_grid, self.current_epoch)

