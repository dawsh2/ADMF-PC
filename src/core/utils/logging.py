"""
Logging utilities for ADMF-PC.

Provides simple logging setup functionality using Python's standard logging.
This replaces the previous structured logging system with a simpler approach.
"""

import logging
import sys
from typing import Optional, List
from pathlib import Path


def setup_logging(
    level: str = 'INFO',
    console: bool = True,
    file_path: Optional[str] = None,
    json_format: bool = False
) -> None:
    """
    Setup basic logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Whether to log to console
        file_path: Optional file path for logging
        json_format: Whether to use JSON format (currently ignored, uses standard format)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if json_format:
        # For now, use a structured-like format instead of actual JSON
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_path:
        log_file = Path(file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def configure_event_logging(event_types: List[str]) -> None:
    """
    Configure logging for specific event types.
    
    This is a simplified version that just logs which event types
    would be enabled. The actual event logging would need to be
    implemented based on the event system.
    
    Args:
        event_types: List of event types to enable logging for
    """
    logger = logging.getLogger(__name__)
    
    if event_types:
        logger.info(f"Event-specific logging would be enabled for: {', '.join(event_types)}")
        # TODO: Implement actual event-specific logging when event system is available
    else:
        logger.debug("No specific event types configured for logging")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)