"""Train GAN model."""
import os

import torch

from src.models.gan.gan import GAN
from src.models.vae.vae import VAE
from src.logger.logger import Logger
from src.logger.gan_tracker import GANTracker as Tracker
from src.data.data_manager import DataManager
from src.models.gan.trainer.gan_trainer import GANTrainer as Trainer
from src.visualization.visualizer import Visualizer
from src.configs import gan_config
from src.configs import vae_config
from src.configs.gan_config import (
    CHANNELS,
    LEARNING_RATE,
    LATENT_DIM,
    IMG_SIZE,
    HIDDEN_DIMS,
    EPOCHS,
    DEVICE,
    NAME,
    LOG_INTERVAL,
    LOGS_PATH,
    SAMPLES_PATH,
)


def main():
    logger = Logger("logs", LOGS_PATH).logger
    logger.info("Logger instantiated")
    tracker = Tracker(NAME)
    logger.info("Tracker instantiated")

    data_manager = DataManager(verbose=True)
    logger.info("Data manager instantiated")

    gan = GAN(
        name=NAME,
        channels=CHANNELS,
        latent_dim=LATENT_DIM,
        img_size=IMG_SIZE,
        hidden_dims=HIDDEN_DIMS,
        logger=logger,
    )

    # Load the vae decoder weights
    # vae = VAE(
    #     name=vae_config.NAME,
    #     channels=vae_config.CHANNELS,
    #     latent_dim=vae_config.LATENT_DIM,
    #     img_size=vae_config.IMG_SIZE,
    #     hidden_dims=vae_config.HIDDEN_DIMS,
    #     logger=logger,
    # )
    # vae.load_from(path=vae_config.MODELS_PATH, name=f"{vae_config.NAME}_best.pt")
    # gan.init_generator(vae.decoder)
    gan.load_from(
        path=os.path.join(
            os.path.dirname(__file__),
            "models_data",
            "SunGAN_D512",
            "models",
        ),
        name="SunGAN_D512_current.pt",
    )

    disc_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=LEARNING_RATE)
    gen_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=10, verbose=True
    # )

    trainer = Trainer(
        model=gan,
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        disc_loss_fn=gan.disc_wgan_loss,
        gen_loss_fn=gan.gen_wgan_loss,
        # scheduler=scheduler,
        is_wgan=True,
        disc_weight_clipping=0.01,
        logger=logger,
        tracker=tracker,
    )
    logger.info("Trainer instantiated")

    visualizer = Visualizer(DEVICE, SAMPLES_PATH)
    logger.info("Visualizer instantiated")

    for param in vars(gan_config):
        logger.info("%s: %s", param, getattr(gan_config, param))
    logger.info("WGAN: %s", trainer.is_wgan)
    logger.info("Discriminator Weight Clipping: %s", trainer.disc_weight_clipping)
    logger.info("%s: %s", "Discriminator Optimizer", disc_optimizer)
    logger.info("%s: %s", "Genenrator Optimizer", gen_optimizer)

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
