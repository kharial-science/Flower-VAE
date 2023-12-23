"""Train VAEs"""
import torch

from src.models.vae.vae import VAE
from src.logger.logger import Logger
from src.logger.tracker import Tracker
from src.data.data_manager import DataManager
from src.trainer.trainer import Trainer
from src.visualization.visualizer import Visualizer
import src.config.config as config
from src.config.config import (
    CHANNELS,
    LEARNING_RATE,
    LATENT_DIM,
    IMG_SIZE,
    KLD_WEIGHT,
    HIDDEN_DIMS,
    EPOCHS,
    DEVICE,
    NAME,
    LOG_INTERVAL,
)


# pylint: disable=missing-function-docstring
def main():
    logger = Logger("logs").logger
    logger.info("Logger instantiated")
    tracker = Tracker(NAME)
    logger.info("Tracker instantiated")

    data_manager = DataManager(verbose=True)
    logger.info("Data manager instantiated")

    vae = VAE(
        name=NAME,
        channels=CHANNELS,
        latent_dim=LATENT_DIM,
        img_size=IMG_SIZE,
        hidden_dims=HIDDEN_DIMS,
        logger=logger,
    )

    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    vae.set_loss_params(kld_weight=KLD_WEIGHT)
    loss_fn = vae.vae_loss

    trainer = Trainer(
        model=vae,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        logger=logger,
        tracker=tracker,
    )
    logger.info("Trainer instantiated")

    visualizer = Visualizer(DEVICE)
    logger.info("Visualizer instantiated")

    for param in vars(config):
        logger.info("%s: %s", param, getattr(config, param))
    logger.info("%s: %s", "loss_fn", loss_fn)
    logger.info("%s: %s", "optimizer", optimizer)
    logger.info("%s: %s", "scheduler", scheduler)

    trainer.train(
        train_dataloader=data_manager.train_dataloader,
        val_dataloader=data_manager.val_dataloader,
        num_epochs=EPOCHS,
        verbose=True,
        visualizer=visualizer,
        log_interval=LOG_INTERVAL,
    )


if __name__ == "__main__":
    main()
