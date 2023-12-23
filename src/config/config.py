"""
Define global variables for the project.
"""

import os

NAME = "Sun_D512_KLD1e-5"

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
GAMMA = 0.9
EPOCHS = 500
KLD_WEIGHT = 1e-5
IMG_SIZE = 128
CHANNELS = 3
VAL_FRAC = 0.1
BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42
LATENT_DIM = 512
HIDDEN_DIMS = [32, 64, 128, 256]
LOG_INTERVAL = 7
FC_LOGVAR_INIT = 1e-6
