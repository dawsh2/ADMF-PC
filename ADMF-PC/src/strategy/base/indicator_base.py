"""
Base implementation for indicators.
"""

from typing import Optional
from datetime import datetime
from abc import ABC, abstractmethod


class IndicatorBase(ABC):
    """Base class for indicators with common functionality."""
    
    def __init__(self, name: str):
        self.name = name
        self._value: Optional[float] = None
        self._is_ready = False
    
    @abstractmethod
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate indicator value."""
        pass
    
    @property
    def value(self) -> Optional[float]:
        """Current indicator value."""
        return self._value
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data."""
        return self._is_ready
    
    def reset(self) -> None:
        """Reset indicator state."""
        self._value = None
        self._is_ready = False