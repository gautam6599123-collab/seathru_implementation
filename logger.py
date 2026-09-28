import logging
import sys


def setup_logger(log_filename="seathru_benchmark.log"):
    """
    Configures a logger to output to both console and file.
    """
    logger = logging.getLogger("SeaThruBenchmark")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the function is called twice
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler (saves to text file)
        file_handler = logging.FileHandler(log_filename, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler (prints to terminal)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
