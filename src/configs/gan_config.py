"""
Define global variables for the GAN
"""

import os

NAME = "SunGAN_D512"

# Define paths and create folders
os.makedirs(
    os.path.join(os.path.dirname(__file__), "..", "..", "models_data", NAME),
    exist_ok=True,
)
os.makedirs(
    os.path.join(os.path.dirname(__file__), "..", "..", "models_data", NAME, "models"),
    exist_ok=True,
)
os.makedirs(
    os.path.join(os.path.dirname(__file__), "..", "..", "models_data", NAME, "logs"),
    exist_ok=True,
)
os.makedirs(
    os.path.join(os.path.dirname(__file__), "..", "..", "models_data", NAME, "samples"),
    exist_ok=True,
)
MODELS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models_data", NAME, "models"
)
LOGS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models_data", NAME, "logs"
)
SAMPLES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models_data", NAME, "samples"
)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "archive", "sunflowers"
)

# Define training parameters
DEVICE = "cpu"
LEARNING_RATE = 1e-4
EPOCHS = 500
IMG_SIZE = 128
CHANNELS = 3
VAL_FRAC = 0.1
BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42
LATENT_DIM = 512
HIDDEN_DIMS = [256, 128, 64, 32]
LOG_INTERVAL = 7
