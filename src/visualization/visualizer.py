"""
Define the visualizer class
"""

import os

import torch
import torchvision.utils as vutils

from src.config.config import SAMPLES_PATH


class Visualizer:
    """
    A class used to visualize generated images from a PyTorch model.

    Attributes
    ----------
    device : str
        The device on which the model is loaded.

    Methods
    -------
    visualize(model, name, num_images=8)
        Generates sample images from the model and saves them as a grid image.
    """

    def __init__(self, device):
        self.device = device

    def save_images(self, name, samples):
        """
        Saves the given images as a grid image.

        Parameters
        ----------
        name : str
            The name of the file to save the generated image as.
        samples : torch.Tensor
            The images to save.
        """
        # Save the grid as an image file
        vutils.save_image(
            samples, os.path.join(SAMPLES_PATH, name + ".png"), nrow=8, normalize=True
        )

    def visualize(self, model, name, num_images=8):
        """
        Generates sample images from the model and saves them as a grid image.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to generate images from.
        name : str
            The name of the file to save the generated image as.
        num_images : int, optional
            The number of images to generate, by default 8.
        """
        # Set model to evaluation mode
        model.eval()

        # Generate sample images from the model
        with torch.no_grad():
            sample = model.sample(num_images)

        # Save the grid as an image file
        self.save_images(name, sample)
