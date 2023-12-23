"""
Define the VAE model
"""

import torch
from torch import nn
from src.models.vae.parts.encoder import Encoder
from src.models.vae.parts.decoder import Decoder


class VAE(nn.Module):
    """
    The VAE model is composed of an encoder and a decoder.
    """

    def __init__(
        self,
        name,
        latent_dim,
        img_size,
        hidden_dims,
        channels=3,
        device="cpu",
        logger=None,
    ):
        super().__init__()
        self.name = name
        self.device = device
        self.logger = logger
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, img_size, channels, hidden_dims)
        hidden_dims.reverse()
        self.decoder = Decoder(latent_dim, channels, hidden_dims, img_size)

        self.kld_weight = None

        logger.info("VAE instantiated")

    def encode(self, x):
        """
        Encodes the input tensor x into a latent space representation.

        Args:
            x (torch.Tensor): Input tensor to encode.

        Returns:
            tuple: A tuple containing the mean and log variance of the encoded tensor.
        """

        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from a Gaussian distribution.

        Args:
            mu (torch.Tensor): Mean of the Gaussian distribution.
            logvar (torch.Tensor): Log variance of the Gaussian distribution.

        Returns:
            torch.Tensor: Sampled tensor from the Gaussian distribution.
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """
        Decodes the latent variable z into the original input space.

        Args:
            z (torch.Tensor): The latent variable to be decoded.

        Returns:
            torch.Tensor: The decoded output.
        """

        x = self.decoder(z)
        return x

    def set_loss_params(self, kld_weight):
        """
        Sets the parameters for the VAE loss.

        Args:
            kld_weight (float): The weight for the Kullback-Leibler divergence loss.
        """

        self.kld_weight = kld_weight

    def vae_loss(self, x_recon, x, mu, logvar):
        """
        Computes the loss for the Variational Autoencoder (VAE) model.

        Args:
            x_recon (torch.Tensor): Reconstructed input tensor of shape (batch_size, input_size).
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            mu (torch.Tensor): Mean tensor of the latent
                variable distribution of shape (batch_size, latent_size).
            logvar (torch.Tensor): Log variance tensor of
                the latent variable distribution of shape (batch_size, latent_size).

        Returns:
            torch.Tensor: The VAE loss.
        """

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="mean")
        # recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="mean")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        loss = recon_loss + kl_loss * self.kld_weight

        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def forward(self, x, verbose=False):
        """
        Forward pass of the Variational Autoencoder (VAE) model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            verbose (bool): Whether to log information about the forward pass.

        Returns:
            x_recon (torch.Tensor): Reconstructed input tensor of shape (batch_size, input_size).
            mu (torch.Tensor): Mean tensor of the latent
                variable distribution of shape (batch_size, latent_size).
            logvar (torch.Tensor): Log variance tensor of
                the latent variable distribution of shape (batch_size, latent_size).
        """

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        if verbose and self.logger is not None:
            self.logger.info(
                "Forward pass completed. Input shape: %s, Output shape: %s",
                x.shape,
                x_recon.shape,
            )
        return x_recon, mu, logvar

    def sample(self, num_samples, noise=None, verbose=False):
        """
        Samples from the latent space and generates output images.

        Args:
            num_samples (int): The number of samples to generate.
            verbose (bool): Whether to log information about the sampling.
            noise (torch.Tensor): The noise to use for sampling.

        Returns:
            torch.Tensor: The generated output images.
        """

        if noise:
            z = noise.to(self.device)
        else:
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
        samples = self.decode(z)
        if verbose and self.logger is not None:
            self.logger.info(
                "Sampling completed. Number of samples: %d, Output shape: %s",
                num_samples,
                samples.shape,
            )
        return samples

    def save_to(self, path, name):
        """
        Saves the model weights to a file.

        Args:
            path (str): The path to save the weights to.
            name (str): The name of the file to save the weights to. Default is "model_weights".
        """

        torch.save(self.state_dict(), f"{path}/{name}")

    def load_from(self, path, name):
        """
        Loads the model weights from a file.

        Args:
            path (str): The path to load the weights from.
            name (str): The name of the file to load the weights from. Default is "model_weights".
        """

        self.load_state_dict(torch.load(f"{path}/{name}"))
