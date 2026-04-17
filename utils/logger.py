"""
Centralized logging setup for the RAG system.
Call setup_logger() once at application startup; all modules then import 'logger'.
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from config import logging_config


def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Configure and return the root logger for the RAG system.

    Args:
        debug: If True, override config and set level to DEBUG.

    Returns:
        Configured Logger instance.
    """
    log_level = logging.DEBUG if debug else getattr(logging, logging_config.level, logging.INFO)

    # Ensure log directory exists
    log_path = Path(logging_config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root logger
    logger = logging.getLogger("rag")
    logger.setLevel(log_level)

    if logger.handlers:
        # Already configured – avoid duplicate handlers
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=logging_config.max_file_size_mb * 1024 * 1024,
        backupCount=logging_config.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# Module-level logger – import this in all modules
logger = setup_logger()
