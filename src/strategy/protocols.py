"""
Strategy module protocols for ADMF-PC.

These protocols define the contracts for all strategy components, enabling
flexible composition without inheritance. Components can implement multiple
protocols to gain different capabilities.
"""

from typing import Protocol, runtime_checkable, Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum


class SignalDirection(Enum):
    """Trading signal direction."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"


@runtime_checkable
class Strategy(Protocol):
    """
    Core protocol for trading strategies.
    
    A strategy processes market data and generates trading signals.
    No inheritance required - any class implementing these methods
    can be used as a strategy.
    """
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal from market data.
        
        Args:
            market_data: Dict containing price data, indicators, etc.
            
        Returns:
            Signal dict with keys:
                - symbol: str
                - direction: SignalDirection
                - strength: float (0-1)
                - timestamp: datetime
                - metadata: Dict[str, Any]
            Or None if no signal
        """
        ...
    
    @property
    def name(self) -> str:
        """Strategy name for identification."""
        ...


@runtime_checkable
class Indicator(Protocol):
    """
    Protocol for technical indicators.
    
    Indicators calculate values from price data that can be used
    by strategies to make trading decisions.
    """
    
    def calculate(self, value: float, timestamp: Optional[datetime] = None) -> Optional[float]:
        """
        Calculate indicator value for new data point.
        
        Args:
            value: New price value
            timestamp: Optional timestamp
            
        Returns:
            Calculated indicator value or None if not ready
        """
        ...
    
    def update(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """
        Update indicator state without returning value.
        More efficient for streaming updates.
        """
        ...
    
    @property
    def value(self) -> Optional[float]:
        """Current indicator value."""
        ...
    
    @property
    def ready(self) -> bool:
        """Whether indicator has enough data to produce values."""
        ...
    
    def reset(self) -> None:
        """Reset indicator state."""
        ...


@runtime_checkable
class Feature(Protocol):
    """
    Protocol for market features derived from multiple data sources.
    
    Features are higher-level market characteristics computed from
    raw data, indicators, or other features.
    """
    
    def extract(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract feature values from market data.
        
        Args:
            data: Market data including prices, indicators, etc.
            
        Returns:
            Dict of feature_name -> feature_value
        """
        ...
    
    @property
    def feature_names(self) -> List[str]:
        """Names of features this extractor produces."""
        ...
    
    def reset(self) -> None:
        """Reset feature extractor state."""
        ...


@runtime_checkable
class Rule(Protocol):
    """
    Protocol for trading rules.
    
    Rules encapsulate specific trading logic that evaluates
    market conditions and produces trading decisions.
    """
    
    def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate rule against market data.
        
        Args:
            data: Market data including prices, indicators, features
            
        Returns:
            Tuple of (is_triggered, signal_strength)
            - is_triggered: Whether rule conditions are met
            - signal_strength: Strength of signal (-1 to 1)
        """
        ...
    
    @property
    def name(self) -> str:
        """Rule name for identification."""
        ...
    
    @property
    def weight(self) -> float:
        """Rule weight for ensemble strategies."""
        ...


@runtime_checkable
class SignalAggregator(Protocol):
    """
    Protocol for combining multiple signals.
    
    Aggregators combine signals from multiple sources (rules, strategies)
    into a single actionable signal.
    """
    
    def aggregate(self, signals: List[Tuple[Dict[str, Any], float]]) -> Optional[Dict[str, Any]]:
        """
        Aggregate multiple weighted signals.
        
        Args:
            signals: List of (signal_dict, weight) tuples
            
        Returns:
            Aggregated signal or None if no consensus
        """
        ...
    
    @property
    def min_signals(self) -> int:
        """Minimum number of signals required."""
        ...


@runtime_checkable
class Classifier(Protocol):
    """
    Protocol for market regime classification.
    
    Classifiers analyze market conditions and assign categorical
    labels (e.g., trending, ranging, volatile).
    """
    
    def classify(self, data: Dict[str, Any]) -> str:
        """
        Classify current market conditions.
        
        Args:
            data: Market data for classification
            
        Returns:
            Classification label
        """
        ...
    
    @property
    def current_class(self) -> Optional[str]:
        """Current classification."""
        ...
    
    @property
    def confidence(self) -> float:
        """Confidence in current classification (0-1)."""
        ...
    
    def reset(self) -> None:
        """Reset classifier state."""
        ...


@runtime_checkable
class RegimeAdaptive(Protocol):
    """
    Protocol for regime-adaptive strategies.
    
    These strategies can modify their behavior based on
    detected market regimes.
    """
    
    def on_regime_change(self, new_regime: str, metadata: Dict[str, Any]) -> None:
        """
        Handle regime change notification.
        
        Args:
            new_regime: New regime classification
            metadata: Additional regime information
        """
        ...
    
    def get_active_parameters(self) -> Dict[str, Any]:
        """Get currently active parameters for current regime."""
        ...


@runtime_checkable
class Optimizable(Protocol):
    """
    Protocol for components that can be optimized.
    
    Any component implementing this protocol can participate
    in optimization workflows.
    """
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dict mapping parameter names to:
                - List[Any]: discrete values
                - Tuple[float, float]: continuous range (min, max)
                - Dict with 'type', 'min', 'max', 'step', etc.
        """
        ...
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameter values."""
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        ...
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameter values.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


@runtime_checkable
class StrategyContainer(Protocol):
    """
    Protocol for strategy containers.
    
    Containers manage strategy lifecycle and provide
    isolation for parallel execution.
    """
    
    def create_strategy(self, spec: Dict[str, Any]) -> Any:
        """Create strategy instance from specification."""
        ...
    
    def initialize_strategy(self, strategy: Any) -> None:
        """Initialize strategy with container context."""
        ...
    
    def reset_strategy(self) -> None:
        """Reset strategy state for new run."""
        ...


@runtime_checkable
class PerformanceTracker(Protocol):
    """
    Protocol for tracking strategy performance metrics.
    
    Used for optimization and analysis of trading results.
    """
    
    def record_trade(self, trade: Dict[str, Any], regime: Optional[str] = None) -> None:
        """Record a completed trade with optional regime context."""
        ...
    
    def get_metrics(self, regime: Optional[str] = None) -> Dict[str, float]:
        """
        Get performance metrics, optionally filtered by regime.
        
        Returns:
            Dict with metrics like sharpe_ratio, win_rate, etc.
        """
        ...
    
    def get_regime_analysis(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics broken down by regime."""
        ...