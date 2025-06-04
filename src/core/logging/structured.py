"""Simple structured logging for unified architecture."""

import logging
from typing import Optional, Dict, Any


class LogContext:
    """Simple logging context."""
    def __init__(self, container_id: str, component_id: str, correlation_id: Optional[str] = None):
        self.container_id = container_id
        self.component_id = component_id
        self.correlation_id = correlation_id


class StructuredLogger:
    """Simple structured logger wrapper."""
    
    def __init__(self, name: str, context: LogContext):
        self.logger = logging.getLogger(name)
        self.context = context
    
    def info(self, message: str, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)