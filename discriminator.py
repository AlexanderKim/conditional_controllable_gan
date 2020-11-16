import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.gen_hidden_block(in_channels, hidden_channels, 4, 2, 1),
            self.gen_hidden_block(hidden_channels, hidden_channels * 2, 4, 2, 1),
            self.gen_hidden_block(hidden_channels * 2, hidden_channels * 4, 4, 2, 1),
            self.gen_hidden_block(hidden_channels * 4, hidden_channels * 8, 4, 2, 1),
            self.gen_output_block(hidden_channels * 8, 1, 4, 1, 0),
        )

    def gen_hidden_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def gen_output_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )

    def forward(self, image):
        return self.disc(image)