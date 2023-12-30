"""
Define a GANTrainer class
"""
import torch

from src.configs.gan_config import MODELS_PATH, LOGS_PATH
from src.visualization.visualizer import Visualizer
from src.logger.logger import Logger
from src.logger.gan_tracker import GANTracker as Tracker
from src.models.gan.gan import GAN


class GANTrainer:
    """
    A class used to train a PyTorch GAN model.

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
    train_step(z, real_images)
        Performs a single training step on the given latent variable.
    train(train_dataloader, val_dataloader, num_epochs, verbose=True)
        Trains the model for the specified number of epochs using the given dataloader.
    eval(dataloader)
        Evaluates the model on the given dataloader.
    """

    def __init__(
        self,
        model: GAN,
        disc_optimizer,
        gen_optimizer,
        disc_loss,
        gen_loss,
        scheduler=None,
        logger: Logger = None,
        tracker: Tracker = None,
    ):
        self.model = model
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.scheduler = scheduler
        self.disc_loss = disc_loss
        self.gen_loss = gen_loss
        self.logger = logger
        self.tracker = tracker

    def train_step(self, z_disc, z_gen, real_images):
        """
        Performs a single training step on the given latent variable.
        """

        # Generate a batch of images from the latent variable z
        generated_images = self.model.generator(z_disc)

        # Forward pass: compute the discriminator's predictions for real and generated images
        real_predictions = self.model.discriminator(real_images)
        generated_predictions = self.model.discriminator(generated_images.detach())

        # Backward pass: compute gradients and update the discriminator's parameters
        self.disc_optimizer.zero_grad()
        disc_loss = self.disc_loss(real_predictions, generated_predictions)
        disc_loss.backward()
        self.disc_optimizer.step()

        # Backward pass of the generator: compute gradients and update the generator's parameters
        generated_images = self.model.generator(z_gen)
        generated_predictions = self.model.discriminator(generated_images)
        self.gen_optimizer.zero_grad()
        gen_loss = self.gen_loss(generated_predictions)
        gen_loss.backward()
        self.gen_optimizer.step()

        return disc_loss, gen_loss

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
        Trains the model for the specified number of epochs using the given dataloader.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader used for training.
            val_dataloader (torch.utils.data.DataLoader): The dataloader used for validation.
            num_epochs (int): The number of epochs to train the model for.
            verbose (bool): Whether to log information about the training.
            visualizer (Visualizer): The visualizer used to visualize the training process.
            log_interval (int): The number of batches to wait before logging training progress.
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

        # Training loop
        # best_val_loss = float("inf")
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            disc_loss = 0
            gen_loss = 0
            for batch_idx, (real_images, _) in enumerate(train_dataloader):
                real_images = real_images.to(self.model.device)

                # Sample random latent variables
                z_disc = torch.randn(real_images.shape[0], self.model.latent_dim).to(
                    self.model.device
                )
                z_gen = torch.randn(real_images.shape[0], self.model.latent_dim).to(
                    self.model.device
                )

                # Perform a single training step
                disc_loss_batch, gen_loss_batch = self.train_step(
                    z_disc, z_gen, real_images
                )
                disc_loss += disc_loss_batch.item()
                gen_loss += gen_loss_batch.item()

                # Log training progress
                if (
                    batch_idx % log_interval == 0
                    and verbose
                    and self.logger is not None
                ):
                    self.logger.info(
                        "Epoch %s/%s, Iteration %s/%s, Discriminator Loss: %.4f, Generator Loss: %.4f",
                        epoch + 1,
                        num_epochs,
                        batch_idx + 1,
                        len(train_dataloader),
                        disc_loss_batch.item(),
                        gen_loss_batch.item(),
                    )

            # Compute average training loss
            disc_loss /= len(train_dataloader.dataset)
            gen_loss /= len(train_dataloader.dataset)

            # Validation
            self.model.eval()
            disc_val_loss = 0
            gen_val_loss = 0
            with torch.no_grad():
                for real_images, _ in val_dataloader:
                    real_images = real_images.to(self.model.device)

                    # Sample random latent variables
                    z = torch.randn(real_images.shape[0], self.model.latent_dim).to(
                        self.model.device
                    )

                    # Generate a batch of images from the latent variable z
                    generated_images = self.model.generator(z)

                    # Forward pass: compute the discriminator's predictions for real and generated images
                    real_predictions = self.model.discriminator(real_images)
                    generated_predictions = self.model.discriminator(generated_images)

                    # Compute the discriminator's loss
                    disc_loss = self.disc_loss(real_predictions, generated_predictions)
                    disc_val_loss += disc_loss.item()

                    # Compute the generator's loss
                    gen_loss = self.gen_loss(generated_predictions)
                    gen_val_loss += gen_loss.item()

            # Compute average validation loss
            disc_val_loss /= len(val_dataloader.dataset)
            gen_val_loss /= len(val_dataloader.dataset)

            # Visualize the model's progress
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

            # Log validation progress
            if verbose and self.logger is not None:
                self.logger.info(
                    "Epoch: %s/%s, Train Disc Loss: %.4f, Train Gen Loss: %.4f, Val Disc Loss: %.4f, Val Gen Loss: %.4f",
                    epoch + 1,
                    num_epochs,
                    disc_loss,
                    gen_loss,
                    disc_val_loss,
                    gen_val_loss,
                )

            self.model.save_to(MODELS_PATH, f"{self.model.name}_current.pt")

            # Update the learning rate
            # if self.scheduler is not None:
            #     self.scheduler.step(disc_val_loss)

            # Update the tracker
            if self.tracker is not None:
                self.tracker.track([disc_loss, gen_loss], [disc_val_loss, gen_val_loss])
                self.tracker.save_to(LOGS_PATH)
                self.tracker.plot(LOGS_PATH)

            # Save the model if it has the best validation loss
            # Gan "Best" is hard to define, therefore this might not be the best way to save the model
            # if disc_val_loss < best_val_loss:
            #     best_val_loss = disc_val_loss
            #     self.model.save_to(MODELS_PATH, f"{self.model.name}_best.pt")
            #     if visualizer is not None:
            #         visualizer.visualize(
            #             self.model,
            #             f"{self.model.name}_large_best",
            #             num_images=64,
            #             noise=vis_noise_large,
            #         )
            #         visualizer.visualize(
            #             self.model,
            #             f"{self.model.name}_small_best",
            #             num_images=64,
            #             noise=vis_noise_small,
            #         )
