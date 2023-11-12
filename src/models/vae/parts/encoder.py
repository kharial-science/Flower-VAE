"""
The Encoder part of the VAE is responsible for encoding the image into a latent vector.
"""

from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    The Encoder module takes an image as input and outputs a latent vector of size latent_dim
    """

    def __init__(self, latent_dim, img_size=64, channels=3, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.img_size = img_size
        self.hidden_dims = hidden_dims.copy()

        self.conv1 = nn.Conv2d(
            channels, hidden_dims[0], kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            hidden_dims[2], hidden_dims[3], kernel_size=3, stride=2, padding=1
        )
        self.fc_mu = nn.Linear(
            hidden_dims[3] * (img_size // 8) * (img_size // 8), latent_dim
        )
        self.fc_logvar = nn.Linear(
            hidden_dims[3] * (img_size // 8) * (img_size // 8), latent_dim
        )

    def forward(self, x):
        """
        Forward pass of the encoder network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            tuple: A tuple of two tensors containing the mean and log variance of the latent space.
        """

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(
            -1, self.hidden_dims[3] * (self.img_size // 8) * (self.img_size // 8)
        )
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
