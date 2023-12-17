"""
Define the Tracker class
"""

import os

import torch
import matplotlib.pyplot as plt


class Tracker:
    """
    A class for tracking the training stats.
    And displaying these stats with matplotlib, and saving the graphs.
    """

    def __init__(self, name):
        self.name = name
        self.losses = []
        self.kld_losses = []
        self.recon_losses = []
        self.val_losses = []

    def track(self, loss, kld_loss, recon_loss, val_loss):
        """
        Track the training stats.

        Args:
            loss (float): The training loss.
            kld_loss (float): The training KLD loss.
            recon_loss (float): The training reconstruction loss.
            val_loss (float): The validation loss.
        """
        self.losses.append(loss)
        self.kld_losses.append(kld_loss)
        self.recon_losses.append(recon_loss)
        self.val_losses.append(val_loss)

    def save_to(self, path):
        """
        Save the training stats to a .pt file.

        Args:
            path (str): The path to save the file.
        """
        file_path = os.path.join(path, f"{self.name}.pt")
        data = {
            "losses": self.losses,
            "kld_losses": self.kld_losses,
            "recon_losses": self.recon_losses,
            "val_losses": self.val_losses,
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
        self.losses = data["losses"]
        self.kld_losses = data["kld_losses"]
        self.recon_losses = data["recon_losses"]
        self.val_losses = data["val_losses"]

    def plot(self, path):
        """
        Plot the training stats and save the graphs.

        Args:
            path (str): The path to save the graphs.
        """
        file_path = os.path.join(path, f"{self.name}")

        # Plot loss curve
        plt.figure(figsize=(12, 8))
        plt.plot(self.losses, label="train loss")
        plt.plot(self.recon_losses, label="train recon loss")
        plt.plot(self.val_losses, label="val loss")
        plt.yscale("log")  # Set y-axis to logarithmic scale
        plt.legend()
        plt.savefig(f"{file_path}_loss.png")
        plt.close()
