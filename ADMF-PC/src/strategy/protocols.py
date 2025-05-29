"""
Core protocols for the strategy module.

These protocols define the contracts that strategy components must implement,
enabling composition and flexibility without inheritance.
"""

from typing import Protocol, runtime_checkable, Dict, Any, Optional, List, Tuple
from abc import abstractmethod
from datetime import datetime


@runtime_checkable
class Indicator(Protocol):
    """Protocol for technical indicators."""
    
    @abstractmethod
    def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
        """Calculate indicator value for new data point."""
        ...
    
    @property
    @abstractmethod
    def value(self) -> Optional[float]:
        """Current indicator value."""
        ...
    
    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether indicator has enough data to produce values."""
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset indicator state."""
        ...


@runtime_checkable
class Feature(Protocol):
    """Protocol for market features derived from indicators."""
    
    @abstractmethod
    def calculate(self, indicators: Dict[str, float]) -> Optional[float]:
        """Calculate feature value from indicator values."""
        ...
    
    @property
    @abstractmethod
    def value(self) -> Optional[float]:
        """Current feature value."""
        ...
    
    @property
    @abstractmethod
    def ready(self) -> bool:
        """Whether feature has enough data."""
        ...


@runtime_checkable
class TradingRule(Protocol):
    """Protocol for trading decision rules."""
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate rule and return (triggered, strength).
        
        Args:
            data: Market data including prices, indicators, features
            
        Returns:
            Tuple of (is_triggered, signal_strength)
        """
        ...
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Rule weight for ensemble strategies."""
        ...


@runtime_checkable
class SignalGenerator(Protocol):
    """Protocol for components that generate trading signals."""
    
    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal from market data.
        
        Returns:
            Signal dict with keys: symbol, direction, strength, timestamp, metadata
        """
        ...


@runtime_checkable
class Strategy(Protocol):
    """Protocol for trading strategies."""
    
    @abstractmethod
    def on_bar(self, bar: Dict[str, Any]) -> None:
        """Process new market bar."""
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Reset strategy state."""
        ...
    
    # Built-in optimization support (with defaults)
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return optimizable parameters. Default: no parameters."""
        return {}
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameters. Default: no-op."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters. Default: empty dict."""
        return {}
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters. Default: always valid."""
        return True, ""


@runtime_checkable
class Classifier(Protocol):
    """Protocol for general classification of market conditions."""
    
    @abstractmethod
    def classify(self, data: Dict[str, Any]) -> str:
        """
        Classify current market conditions.
        
        Args:
            data: Dict containing indicator values or other features
            
        Returns:
            Classification label (e.g., 'TRENDING_UP', 'HIGH_VOLATILITY', 'BULLISH', etc.)
        """
        ...
    
    @property
    @abstractmethod
    def current_class(self) -> Optional[str]:
        """Current classification."""
        ...
    
    @property
    @abstractmethod
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        ...


@runtime_checkable
class MetaLabeler(Protocol):
    """Protocol for signal quality assessment."""
    
    @abstractmethod
    def evaluate_signal(self, signal: Dict[str, Any], 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate signal quality and add metadata.
        
        Returns:
            Enhanced signal with quality metrics
        """
        ...


@runtime_checkable
class SignalAggregator(Protocol):
    """Protocol for combining multiple signals."""
    
    @abstractmethod
    def aggregate(self, signals: List[Tuple[Dict[str, Any], float]]) -> Optional[Dict[str, Any]]:
        """
        Aggregate multiple weighted signals into one.
        
        Args:
            signals: List of (signal, weight) tuples
            
        Returns:
            Aggregated signal or None
        """
        ...