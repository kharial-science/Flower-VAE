import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.vae.parts.decoder import Decoder
from src.models.gan.parts.discriminator import Discriminator


class GAN(nn.Module):
    """
    The GAN model is composed of a generator and a discriminator.
    It is used to fine tune the decoder of the VAE.
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
        self.generator = Decoder(latent_dim, channels, hidden_dims, img_size)
        hidden_dims.reverse()
        self.discriminator = Discriminator(img_size, channels, hidden_dims)

        logger.info("GAN instantiated")

    def generate(self, z):
        """
        Generates an image from a latent space representation.

        Args:
            z (torch.Tensor): Latent space representation.

        Returns:
            torch.Tensor: Generated image.
        """

        return self.generator(z)

    def discriminate(self, x):
        """
        Discriminates an image.

        Args:
            x (torch.Tensor): Image to discriminate.

        Returns:
            torch.Tensor: Discrimination score.
        """

        return self.discriminator(x)

    def forward(self, z):
        """
        Forward pass of the GAN network.
        """

        return self.discriminate(self.generate(z))

    def disc_std_loss(self, real_predictions, generated_predictions):
        """
        Computes the GAN loss.

        Args:
            real_predictions (torch.Tensor): Discriminator's predictions for real images.
            generated_predictions (torch.Tensor): Discriminator's predictions for generated images.

        Returns:
            torch.Tensor: GAN loss.
        """

        real_loss = F.binary_cross_entropy_with_logits(
            real_predictions, torch.ones_like(real_predictions)
        )
        generated_loss = F.binary_cross_entropy_with_logits(
            generated_predictions, torch.zeros_like(generated_predictions)
        )
        return real_loss + generated_loss

    def gen_std_loss(self, generated_predictions):
        """
        Computes the generator loss.

        Args:
            generated_predictions (torch.Tensor): Discriminator's predictions for generated images.

        Returns:
            torch.Tensor: Generator loss.
        """

        return F.binary_cross_entropy_with_logits(
            generated_predictions, torch.ones_like(generated_predictions)
        )

    def disc_wgan_loss(self, real_predictions, generated_predictions):
        """
        Computes the GAN loss.

        Args:
            real_predictions (torch.Tensor): Discriminator's predictions for real images.
            generated_predictions (torch.Tensor): Discriminator's predictions for generated images.

        Returns:
            torch.Tensor: GAN loss.
        """

        real_loss = real_predictions.mean()
        generated_loss = generated_predictions.mean()
        return -real_loss + generated_loss

    def gen_wgan_loss(self, generated_predictions):
        """
        Computes the generator loss.

        Args:
            generated_predictions (torch.Tensor): Discriminator's predictions for generated images.

        Returns:
            torch.Tensor: Generator loss.
        """

        return -generated_predictions.mean()

    def sample(self, num_samples, noise=None, verbose=False):
        """
        Samples from the latent space and generates output images.

        Args:
            num_samples (int): The number of samples to generate.
            noise (torch.Tensor): Noise tensor to sample from.
            verbose (bool): Whether to log information about the forward pass.

        Returns:
            torch.Tensor: Generated images.
        """

        if noise is None:
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
        samples = self.generate(noise)
        if verbose and self.logger is not None:
            self.logger.info(
                "Sampling completed. Input shape: %s, Output shape: %s",
                noise.shape,
                samples.shape,
            )
        return samples

    def init_generator(self, model: nn.Module):
        """
        Initializes the generator with the weights of the given model.
        The weights of the discriminator are not affected.
        The weights of the generator are supposed to be the weights of a trained VAE.

        Args:
            model (nn.Module): Model to initialize the generator with.
        """

        self.generator.load_state_dict(model.state_dict())

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
