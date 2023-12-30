"""
The Encoder part of the VAE is responsible for encoding the image into a latent vector.
"""

from torch import nn
import torch.nn.functional as F

from src.configs.vae_config import FC_LOGVAR_INIT


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
        self.bn1 = nn.BatchNorm2d(hidden_dims[0])
        self.conv2 = nn.Conv2d(
            hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])
        self.conv3 = nn.Conv2d(
            hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1
        )
        self.bn3 = nn.BatchNorm2d(hidden_dims[2])
        self.conv4 = nn.Conv2d(
            hidden_dims[2], hidden_dims[3], kernel_size=3, stride=2, padding=1
        )
        self.bn4 = nn.BatchNorm2d(hidden_dims[3])
        self.fc_mu = nn.Linear(
            hidden_dims[3] * (img_size // 8) * (img_size // 8), latent_dim
        )
        self.fc_logvar = nn.Linear(
            hidden_dims[3] * (img_size // 8) * (img_size // 8), latent_dim
        )
        # Initialize fc_logvar with small weights, to avoid kld divergence with batch norm
        # With the presence of batch norm, kld loss might explode because of this exponential term
        # The weights are initialized with a small value to avoid this,
        # and the learning rate should be set accordingly
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=FC_LOGVAR_INIT)
        nn.init.constant_(self.fc_logvar.bias, 0)

    def forward(self, x):
        """
        Forward pass of the encoder network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            tuple: A tuple of two tensors containing the mean and log variance of the latent space.
        """

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(
            -1, self.hidden_dims[3] * (self.img_size // 8) * (self.img_size // 8)
        )
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
