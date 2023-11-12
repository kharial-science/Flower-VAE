"""
Define global variables for the project.
"""

import os

# Define paths
MODELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models")
LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "samples")

# Define training parameters
DEVICE = "cpu"
LEARNING_RATE = 5e-4
GAMMA = 0.95
EPOCHS = 50
KLD_WEIGHT = 1e-3
NAME = "FlowerVAE"
IMG_SIZE = 64
CHANNELS = 3
VAL_FRAC = 0.1
BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42
LATENT_DIM = 128
HIDDEN_DIMS = [32, 64, 128, 256]
