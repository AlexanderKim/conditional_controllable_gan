import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noize_dim=10, output_dim=3, hidden_dim=64):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self.gen_hidden_block(noize_dim, hidden_dim * 8, 4, 1, 0),
            self.gen_hidden_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            self.gen_hidden_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            self.gen_hidden_block(hidden_dim * 2, hidden_dim, 4, 2, 1),
            self.gen_output_block(hidden_dim, output_dim, 4, 2, 1)
        )

        self.noize_dim = noize_dim

    def gen_hidden_block(self, input_size, output_size, kernel_size=4, stride=1, padding=0, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=out_padding),
            nn.BatchNorm2d(output_size),
            nn.ReLU()
        )

    def gen_output_block(self, input_size, output_size, kernel_size=4, stride=1, padding=0, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_size, out_channels=output_size, kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=out_padding),
            nn.Tanh()
        )

    def forward(self, noize):
        return self.gen(noize)

    def gen_noize(self, n_samples=128, device='cuda'):
        noize = torch.randn(n_samples, self.noize_dim, device=device)
        return noize.view(n_samples, self.noize_dim, 1, 1)