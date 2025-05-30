"""
Data protocols for ADMF-PC using typing.Protocol (NO ABC inheritance).

This module defines pure behavioral contracts without implementation,
following the Protocol+Composition architecture.
"""

from typing import Protocol, runtime_checkable, List, Optional, Dict, Any, Iterator
from datetime import datetime
import pandas as pd

from .models import Bar


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for components that provide market data - NO INHERITANCE!"""
    
    def load_data(self, symbols: List[str]) -> bool:
        """Load data for specified symbols"""
        ...
    
    def get_symbols(self) -> List[str]:
        """Get loaded symbols"""
        ...
    
    def has_data(self, symbol: str) -> bool:
        """Check if data exists for symbol"""
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
