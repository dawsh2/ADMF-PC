"""
Data protocols for ADMF-PC using typing.Protocol (NO ABC inheritance).

This module defines pure behavioral contracts without implementation,
following the Protocol+Composition architecture.
"""

from typing import Protocol, runtime_checkable, List, Optional, Dict, Any, Iterator, Callable, TYPE_CHECKING
from datetime import datetime
import pandas as pd

from .models import Bar

# Type alias for event handlers
EventHandler = Callable[[Any], None]

# Import Event for type hints
if TYPE_CHECKING:
    from ..core.events.types import Event
else:
    Event = Any


# Define DataProvider protocol here since data module owns it
@runtime_checkable
class DataProvider(Protocol):
    """
    Unified protocol for market data providers.
    
    Combines historical data retrieval and real-time subscription capabilities.
    Used by both data module and components that need data access.
    """
    
    def get_data(self, symbol: str, start: datetime, end: datetime) -> Any:
        """Retrieve historical market data for a symbol and time range."""
        ...
    
    def load_data(self, symbols: List[str]) -> bool:
        """Load data for specified symbols."""
        ...
    
    def get_symbols(self) -> List[str]:
        """Get list of loaded symbols."""
        ...
    
    def subscribe_symbol(self, symbol: str, handler: EventHandler) -> None:
        """Subscribe to real-time updates for a symbol."""
        ...
    
    def unsubscribe_symbol(self, symbol: str, handler: EventHandler) -> None:
        """Unsubscribe from symbol updates."""
        ...


@runtime_checkable
class HistoricalDataProvider(Protocol):
    """
    Specialized protocol for historical data only.
    
    For cases where real-time capabilities are not needed.
    """
    
    def get_data(self, symbol: str, start: datetime, end: datetime) -> Any:
        """Retrieve historical market data for a symbol and time range."""
        ...
        
    def get_symbols(self) -> List[str]:
        """Get available symbols."""
        ...


@runtime_checkable
class RealtimeDataProvider(Protocol):
    """
    Specialized protocol for real-time data only.
    
    For cases where historical data is not needed.
    """
    
    def subscribe_symbol(self, symbol: str, handler: EventHandler) -> None:
        """Subscribe to real-time updates for a symbol."""
        ...
    
    def unsubscribe_symbol(self, symbol: str, handler: EventHandler) -> None:
        """Unsubscribe from symbol updates."""
        ...
        
    def get_subscribed_symbols(self) -> List[str]:
        """Get currently subscribed symbols."""
        ...


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for loading market data from sources - NO INHERITANCE!"""
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load market data for a symbol"""
        ...
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate loaded data"""
        ...


@runtime_checkable
class BarStreamer(Protocol):
    """Protocol for streaming market data bars - NO INHERITANCE!"""
    
    def update_bars(self) -> bool:
        """Advance to next bar and emit events"""
        ...
    
    def has_more_data(self) -> bool:
        """Check if more data is available"""
        ...
    
    def reset(self) -> None:
        """Reset to beginning of data"""
        ...


@runtime_checkable
class SignalStreamer(Protocol):
    """Protocol for streaming stored signals - NO INHERITANCE!"""
    
    def stream_signals(self) -> Iterator['Event']:
        """Stream signal events from storage"""
        ...
    
    def has_more_signals(self) -> bool:
        """Check if more signals are available"""
        ...
    
    def reset(self) -> None:
        """Reset to beginning of signal stream"""
        ...


@runtime_checkable
class DataAccessor(Protocol):
    """Protocol for accessing historical data - NO INHERITANCE!"""
    
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get most recent bar for symbol"""
        ...
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Bar]:
        """Get last N bars for symbol"""
        ...
    
    def get_bar_at_index(self, symbol: str, index: int) -> Optional[Bar]:
        """Get bar at specific index"""
        ...


@runtime_checkable
class DataSplitter(Protocol):
    """Protocol for train/test data splitting - NO INHERITANCE!"""
    
    def setup_split(self, method: str = 'ratio', **kwargs) -> None:
        """Set up train/test split"""
        ...
    
    def set_active_split(self, split_name: Optional[str]) -> None:
        """Set active data split (train/test/None)"""
        ...
    
    def get_split_info(self) -> Dict[str, Any]:
        """Get information about current splits"""
        ...


@runtime_checkable
class DataValidator(Protocol):
    """Protocol for data validation - NO INHERITANCE!"""
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data and return validation results"""
        ...
    
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rule names"""
        ...


@runtime_checkable
class DataTransformer(Protocol):
    """Protocol for data transformation - NO INHERITANCE!"""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        ...
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverse transformation"""
        ...


@runtime_checkable
class StreamingProvider(Protocol):
    """Protocol for real-time data streaming - NO INHERITANCE!"""
    
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols"""
        ...
    
    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols"""
        ...
    
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get latest bar from stream"""
        ...


@runtime_checkable
class DataFeed(Protocol):
    """Protocol for unified data access (historical + streaming) - NO INHERITANCE!"""
    
    def get_current_bar(self, symbol: str) -> Optional[Bar]:
        """Get current bar data"""
        ...
    
    def advance(self) -> bool:
        """Advance to next time step"""
        ...
    
    def get_historical_data(self, symbol: str, lookback_bars: int) -> pd.DataFrame:
        """Get historical data up to current point"""
        ...


# Event-related protocols

@runtime_checkable  
class EventEmitter(Protocol):
    """Protocol for components that emit events - NO INHERITANCE!"""
    
    def emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit an event"""
        ...


@runtime_checkable
class EventSubscriber(Protocol):
    """Protocol for components that subscribe to events - NO INHERITANCE!"""
    
    def subscribe_to_event(self, event_type: str, handler: callable) -> None:
        """Subscribe to an event type"""
        ...
    
    def unsubscribe_from_event(self, event_type: str, handler: callable) -> None:
        """Unsubscribe from an event type"""
        ...


# Capability detection protocols

@runtime_checkable
class HasLifecycle(Protocol):
    """Protocol for components with lifecycle management - NO INHERITANCE!"""
    
    def start(self) -> None:
        """Start the component"""
        ...
    
    def stop(self) -> None:
        """Stop the component"""
        ...
    
    def reset(self) -> None:
        """Reset the component"""
        ...


@runtime_checkable
class HasLogging(Protocol):
    """Protocol for components with logging capability - NO INHERITANCE!"""
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log info message"""
        ...
    
    def log_error(self, message: str, **kwargs) -> None:
        """Log error message"""
        ...


@runtime_checkable
class HasMonitoring(Protocol):
    """Protocol for components with monitoring capability - NO INHERITANCE!"""
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        ...
    
    def record_metric(self, name: str, value: Any) -> None:
        """Record a metric value"""
        ...
