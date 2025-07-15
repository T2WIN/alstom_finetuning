# src/utils/logging_setup.py

import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logging(output_folder: str, log_level: str = "INFO"):
    """
    Configures logging to output to both the console and a rotating file.

    This function sets up a root logger. It ensures that log messages are
    formatted consistently and are sent to the appropriate destinations
    based on their severity level.

    Args:
        output_folder (str): The path to the folder where the log file will be stored.
        log_level (str): The minimum level of logs to display on the console (e.g., "INFO", "DEBUG").
    """
    # Ensure the output directory for logs exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture everything

    # Prevent propagation to avoid duplicate logs if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console Handler ---
    # This handler prints logs to the console (standard output).
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set the level for the console. e.g., INFO will show INFO, WARNING, ERROR, CRITICAL
    try:
        console_log_level = getattr(logging, log_level.upper())
        console_handler.setLevel(console_log_level)
    except AttributeError:
        print(f"Invalid log level '{log_level}'. Defaulting to INFO.")
        console_handler.setLevel(logging.INFO)

    # Define the format for console messages
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # --- File Handler ---
    # This handler saves logs to a file.
    log_file_path = os.path.join(output_folder, "pipeline.log")
    
    # RotatingFileHandler ensures log files don't grow indefinitely.
    # maxBytes=5MB, keeping up to 5 backup files.
    file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=2)
    
    # The file handler should capture all details, so we set it to DEBUG.
    file_handler.setLevel(logging.DEBUG)
    
    # Define a more detailed format for file logs
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)-20s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Add both handlers to the root logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"Logging configured. Console level: {log_level.upper()}, File level: DEBUG.")
    logging.info(f"Log file located at: {log_file_path}")

