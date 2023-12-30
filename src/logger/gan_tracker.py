import os

import torch
import matplotlib.pyplot as plt


class GANTracker:
    """
    A class for tracking the training stats of the GAN.
    And displaying these stats with matplotlib, and saving the graphs.
    """

    def __init__(self, name):
        self.name = name
        self.disc_losses = []
        self.gen_losses = []

    def track(self, loss):
        """
        Track the training stats.

        Args:
            loss (tuple): The training loss.
            val_loss (tuple): The validation loss.
        """
        self.disc_losses.append(loss[0])
        self.gen_losses.append(loss[1])

    def save_to(self, path):
        """
        Save the training stats to a .pt file.

        Args:
            path (str): The path to save the file.
        """
        file_path = os.path.join(path, f"{self.name}.pt")
        data = {
            "losses": (self.disc_losses, self.gen_losses),
        }
        torch.save(data, file_path)

    def load_from(self, path):
        """
        Load the training stats from a .pt file.

        Args:
            path (str): The path to load the file from.
        """
        file_path = os.path.join(path, f"{self.name}.pt")
        data = torch.load(file_path)
        self.disc_losses, self.gen_losses = data["losses"]

    def plot(self, path):
        """
        Plot the training stats.

        Args:
            path (str): The path to save the plot to.
        """
        file_path = os.path.join(path, f"{self.name}")

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(self.gen_losses, label="generator_loss")
        ax.plot(self.disc_losses, label="discriminator_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("GAN Training Loss")
        ax.legend()
        fig.savefig(f"{file_path}_loss.png")
        plt.close()
