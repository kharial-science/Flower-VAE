from torch import nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    The Discriminator module takes an image as input and
    outputs a probability of the image being real.
    """

    def __init__(self, img_size, channels, hidden_dims):
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.channels = channels
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
        self.fc1 = nn.Linear(
            hidden_dims[3] * (img_size // 8) * (img_size // 8), hidden_dims[3]
        )
        self.fc2 = nn.Linear(hidden_dims[3], 1)

    def forward(self, x):
        """
        Forward pass of the discriminator network.
        """

        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        out = F.leaky_relu(self.bn3(self.conv3(out)), 0.2)
        out = F.leaky_relu(self.bn4(self.conv4(out)), 0.2)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.fc1(out), 0.2)
        out = self.fc2(out)
        return out
