"""
Define global variables for the project.
"""

import os

# Define paths
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models")
LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "archive", "sunflowers"
)
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "samples")

# Define training parameters
DEVICE = "cpu"
LEARNING_RATE = 1e-3
GAMMA = 0.95
EPOCHS = 100
KLD_WEIGHT = 1e-3
NAME = "FlowerVAE"
IMG_SIZE = 128
CHANNELS = 3
VAL_FRAC = 0.1
BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42
LATENT_DIM = 128
HIDDEN_DIMS = [32 * 2, 64 * 2, 128 * 2, 256 * 2]
