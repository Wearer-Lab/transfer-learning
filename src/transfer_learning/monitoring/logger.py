"""Centralized logging configuration for the video processing pipeline."""

import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from typing import Optional

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Name of the logger
        log_file: Optional path to log file
        level: Logging level
        format: Log message format
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create console handler with rich formatting
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    
    # Add file handler if log_file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(file_handler)
    
    return logger 