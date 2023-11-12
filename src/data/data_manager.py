"""
Define the DataManager class
"""

import torch

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose,
    ToTensor,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)

from src.config.config import (
    VAL_FRAC,
    DATA_PATH,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    IMG_SIZE,
)


class DataManager:
    """
    A class used to manage the data for a machine learning project.

    Attributes
    ----------
    transform : torchvision.transforms.Compose
        A composition of image transformations to be applied to the dataset.
    dataset : torchvision.datasets.ImageFolder
        The dataset containing the images and their labels.
    train_dataset : torch.utils.data.Subset
        The subset of the dataset used for training.
    val_dataset : torch.utils.data.Subset
        The subset of the dataset used for validation.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader used for training.
    val_dataloader : torch.utils.data.DataLoader
        The dataloader used for validation.

    Methods
    -------
    _instantiate_datasets():
        Instantiates the train and validation datasets from the data folder.
    _instantiate_dataloaders():
        Instantiates the train and validation dataloaders for the given datasets.
    """

    def __init__(self, verbose=False, logger=None):
        self.verbose = verbose
        self.logger = logger

        self.transform = None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

        self._instantiate_datasets()
        self._instantiate_dataloaders()

        if verbose and logger is not None:
            self.logger.info("Training dataset length: %d", len(self.train_dataset))
            self.logger.info("Validation dataset length: %d", len(self.val_dataset))

    def _instantiate_datasets(self):
        """
        Instantiates the train and validation datasets from the data folder.
        """
        if self.verbose and self.logger is not None:
            self.logger.info("Instantiating datasets...")
        self.transform = Compose(
            [
                # RandomRotation(degrees=90),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                Resize((IMG_SIZE, IMG_SIZE)),
                ToTensor(),
            ]
        )
        self.dataset = ImageFolder(root=DATA_PATH, transform=self.transform)
        total_length = len(self.dataset)
        val_length = int(total_length * VAL_FRAC)
        train_length = total_length - val_length
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_length, val_length],
            generator=torch.Generator().manual_seed(SEED),
        )
        if self.verbose and self.logger is not None:
            self.logger.info("Datasets instantiated.")

    def _instantiate_dataloaders(self):
        """
        Instantiates the train and validation dataloaders for the given datasets.
        """
        if self.verbose and self.logger is not None:
            self.logger.info("Instantiating dataloaders...")
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if self.verbose and self.logger is not None:
            self.logger.info("Dataloaders instantiated.")
