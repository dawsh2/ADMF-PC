"""
Feature protocols and interfaces for incremental computation.

Defines protocols that all incremental features must implement.
No inheritance - pure protocol + composition architecture.
"""

from typing import Protocol, Optional, Any, Dict


class Feature(Protocol):
    """
    Protocol for incremental features that maintain state.
    
    All features update incrementally with O(1) complexity
    instead of recomputing from scratch.
    """
    
    @property
    def name(self) -> str:
        """Feature name identifier."""
        ...
    
    @property
    def value(self) -> Optional[Any]:
        """Get current feature value if ready."""
        ...
    
    @property
    def is_ready(self) -> bool:
        """Check if feature has enough data to produce valid values."""
        ...
    
    def update(self, price: float, high: Optional[float] = None, 
               low: Optional[float] = None, volume: Optional[float] = None) -> Optional[Any]:
        """Update feature with new price data and return current value."""
        ...
    
    def reset(self) -> None:
        """Reset feature state."""
        ...


class FeatureState:
    """
    Composable feature state management.
    
    Provides common state tracking for features.
    """
    
    def __init__(self, name: str):
        self.name = name
        self._value: Optional[Any] = None
        self._is_ready = False
    
    @property
    def value(self) -> Optional[Any]:
        """Get current feature value if ready."""
        return self._value if self._is_ready else None
    
    @property
    def is_ready(self) -> bool:
        """Check if feature has enough data to produce valid values."""
        return self._is_ready
    
    def set_value(self, value: Any) -> None:
        """Set feature value and mark as ready."""
        self._value = value
        self._is_ready = True
    
    def reset(self) -> None:
        """Reset feature state."""
        self._value = None
        self._is_ready = False