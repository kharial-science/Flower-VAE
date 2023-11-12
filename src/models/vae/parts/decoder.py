"""
The Decoder part of the VAE is responsible for reconstructing the image from the latent vector.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    The Decoder module takes a latent vector as input and outputs an image
    """

    def __init__(self, latent_dim, channels=3, hidden_dims=None, img_size=64):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.hidden_dims = hidden_dims
        self.img_size = img_size

        self.fc = nn.Linear(
            latent_dim, hidden_dims[0] * (img_size // 8) * (img_size // 8)
        )
        self.conv1 = nn.ConvTranspose2d(
            hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.ConvTranspose2d(
            hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.ConvTranspose2d(
            hidden_dims[2], hidden_dims[3], kernel_size=4, stride=2, padding=1
        )
        self.conv4 = nn.ConvTranspose2d(
            hidden_dims[3], channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size=input_size).
        """

        x = self.fc(z)
        x = x.view(-1, self.hidden_dims[0], 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        x = x.view(-1, 3, self.img_size, self.img_size)

        return x
