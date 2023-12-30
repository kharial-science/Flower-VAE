"""
Define a trainer class
"""

import torch

from src.configs.vae_config import MODELS_PATH, LOGS_PATH
from src.visualization.visualizer import Visualizer
from src.logger.logger import Logger
from src.logger.vae_tracker import VAETracker as Tracker


class VAETrainer:
    """
    A class used to train a PyTorch VAE model.

    Attributes
    ----------
    model : torch.nn.Module
        The PyTorch model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model's parameters.
    loss_fn : callable
        The loss function used to compute the loss.
    scheduler : torch.optim.lr_scheduler._LRScheduler
        The learning rate scheduler used to adjust the learning rate during training.

    Methods
    -------
    train_step(x, y)
        Performs a single training step on the given input and target tensors.
    train(train_dataloader, val_dataloader, num_epochs, verbose=True)
        Trains the model for the specified number of epochs using the given dataloader.
    eval(dataloader)
        Evaluates the model on the given dataloader.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        scheduler=None,
        logger: Logger = None,
        tracker: Tracker = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.logger = logger
        self.tracker = tracker

    def train_step(self, x):
        """
        Performs a single training step on the given input and target tensors.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            y (torch.Tensor): Target tensor of shape (batch_size, output_size).

        Returns:
            float: The loss value for this training step.
        """
        # Set model to training mode
        self.model.train()

        # Forward pass
        x_recon, mu, logvar = self.model(x)

        # Compute loss
        loss = self.loss_fn(x_recon, x, mu, logvar)

        # Zero gradients
        self.optimizer.zero_grad()

        # Backward pass
        loss["loss"].backward()

        # Update weights
        self.optimizer.step()

        return loss

    def train(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs,
        verbose=True,
        visualizer: Visualizer = None,
        log_interval=10,
    ):
        """
        Trains the model for a specified number of epochs using the given dataloader.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader to use for training.
            val_dataloader (torch.utils.data.DataLoader): The dataloader to use for validation.
            num_epochs (int): The number of epochs to train the model for.
            verbose (bool): Whether to print the epoch and loss information. Defaults to True.
        """

        # Define visualization parameters
        vis_num_samples = 64
        vis_spread_large = 10
        vis_noise_large = (
            torch.randn(vis_num_samples, self.model.latent_dim) * vis_spread_large
        )
        vis_spread_small = 1
        vis_noise_small = (
            torch.randn(vis_num_samples, self.model.latent_dim) * vis_spread_small
        )

        if visualizer is not None:
            visualizer.visualize(
                self.model,
                f"{self.model.name}_large_init",
                num_images=64,
                noise=vis_noise_large,
            )
            visualizer.visualize(
                self.model,
                f"{self.model.name}_small_init",
                num_images=64,
                noise=vis_noise_small,
            )

        # Run the training loop
        best_val_loss = float("inf")
        for epoch in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            for i, data in enumerate(train_dataloader):
                if i == 0 and visualizer is not None:
                    visualizer.save_images("real_samples", data[0])
                loss = self.train_step(data[0])
                if i % log_interval == 0:
                    if verbose and self.logger is not None:
                        self.logger.info(
                            "Epoch %s/%s, Iteration %s/%s, Loss: %.4f, ReconsLoss: %.4f, KldLoss: %.4f",
                            epoch + 1,
                            num_epochs,
                            i + 1,
                            len(train_dataloader),
                            loss["loss"].item(),
                            loss["recon_loss"].item(),
                            loss["kl_loss"].item(),
                        )
                train_loss += loss["loss"].item()
            train_loss /= len(train_dataloader)
            val_loss = self.eval(val_dataloader)
            if visualizer is not None:
                visualizer.visualize(
                    self.model,
                    f"{self.model.name}_large_epoch_{epoch}",
                    num_images=64,
                    noise=vis_noise_large,
                )
                visualizer.visualize(
                    self.model,
                    f"{self.model.name}_small_epoch_{epoch}",
                    num_images=64,
                    noise=vis_noise_small,
                )
            if verbose and self.logger is not None:
                self.logger.info(
                    "Epoch %s/%s, Train Loss: %.4f, Val Loss: %.4f",
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    val_loss,
                )
            self.model.save_to(MODELS_PATH, f"{self.model.name}_current.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_to(MODELS_PATH, f"{self.model.name}_best.pt")
                visualizer.visualize(
                    self.model,
                    "best_samples_large",
                    num_images=64,
                    noise=vis_noise_large,
                )
                visualizer.visualize(
                    self.model,
                    "best_samples_small",
                    num_images=64,
                    noise=vis_noise_small,
                )
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            if self.tracker is not None:
                self.tracker.track(
                    train_loss,
                    loss["kl_loss"].item(),
                    loss["recon_loss"].item(),
                    val_loss,
                )
                self.tracker.save_to(LOGS_PATH)
                self.tracker.plot(LOGS_PATH)

    def eval(self, dataloader):
        """
        Evaluates the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to use for evaluation.

        Returns:
            float: The average loss value for the given dataloader.
        """

        loss = 0.0
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                self.model.eval()
                x_recon, mu, logvar = self.model(data[0])
                loss += self.loss_fn(x_recon, data[0], mu, logvar)["loss"].item()
        loss /= len(dataloader)
        return loss
