"""
Base implementation for signal generators.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class SignalGeneratorBase(ABC):
    """Base class for signal generators."""
    
    def __init__(self, name: str):
        self.name = name
        self.signal_count = 0
        self.last_signal: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trading signal from data."""
        pass
    
    def reset(self) -> None:
        """Reset generator state."""
        self.signal_count = 0
        self.last_signal = None