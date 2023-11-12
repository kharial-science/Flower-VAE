"""
Define the Logger class
"""

import logging
from src.config.config import LOGS_PATH


class Logger:
    """
    A class for creating and configuring a logger object.

    Attributes:
        logger (logging.Logger): The logger object.
    """

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(LOGS_PATH + f"/{name}.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
