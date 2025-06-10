"""
Data handlers using Protocol+Composition - NO INHERITANCE!

Simple classes that implement data protocols through duck typing.
No inheritance - protocols are implemented directly.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path
import logging

from .models import Bar, Timeframe, DataSplit
from .loaders import SimpleCSVLoader

logger = logging.getLogger(__name__)


class SimpleHistoricalDataHandler:
    """
    Simple historical data handler - NO INHERITANCE!
    Implements multiple protocols through duck typing.
    """
    
    def __init__(self, handler_id: str = "historical_data", data_dir: str = "data"):
        # Simple initialization - no base class complexity
        self.handler_id = handler_id
        self.data_dir = data_dir
        
        # Data storage
        self.symbols: List[str] = []
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_indices: Dict[str, int] = {}
        
        # Train/test splits
        self.splits: Dict[str, DataSplit] = {}
        self.active_split: Optional[str] = None
        
        # Timeline for multi-symbol synchronization
        self._timeline: List[Tuple[datetime, str]] = []
        self._timeline_idx = 0
        
        # State
        self._running = False
        
        # Data loader
        self.loader = SimpleCSVLoader(data_dir)
        
        # Event bus - set by container
        self.event_bus = None
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return self.handler_id
    
    def set_event_bus(self, event_bus) -> None:
        """Set the event bus for publishing events."""
        self.event_bus = event_bus
    
    # Implements DataProvider protocol
    def load_data(self, symbols: List[str]) -> bool:
        """Load data for specified symbols."""
        self.symbols = symbols
        
        try:
            for symbol in symbols:
                # Load data using loader
                df = self.loader.load(symbol)
                
                # Store data
                self.data[symbol] = df
                self.current_indices[symbol] = 0
            
            # Build synchronized timeline
            self._build_timeline()
            return True
            
        except Exception as e:
            if hasattr(self, 'log_error'):  # If logging capability added
                self.log_error(f"Failed to load data: {e}")
            return False
    
    def get_symbols(self) -> List[str]:
        """Get loaded symbols."""
        return self.symbols.copy()
    
    def has_data(self, symbol: str) -> bool:
        """Check if data exists for symbol."""
        return symbol in self.data
    
    def execute(self) -> None:
        """
        Execute data streaming - called during container execution phase.
        
        Streams all available bars through the event system.
        This enables event-driven execution where other components
        react naturally to BAR events.
        """
        if not self._running:
            self.start()
        
        # Stream all bars
        bars_streamed = 0
        max_bars = getattr(self, 'max_bars', float('inf'))
        
        while self.has_more_data() and bars_streamed < max_bars:
            if self.update_bars():
                bars_streamed += 1
            else:
                break
        
        logger.info(f"Data handler streamed {bars_streamed} bars")
    
    # Implements BarStreamer protocol
    def update_bars(self) -> bool:
        """Advance to next bar and emit events."""
        if not self._running:
            return False
        
        # Get active dataset
        data_dict = self._get_active_data()
        indices = self._get_active_indices()
        
        # Get timeline for current split
        if self.active_split:
            timeline = self._build_split_timeline(data_dict)
        else:
            timeline = self._timeline
        
        if self._timeline_idx >= len(timeline):
            return False
        
        # Get next bar
        timestamp, symbol = timeline[self._timeline_idx]
        self._timeline_idx += 1
        
        # Get bar data
        symbol_data = data_dict[symbol]
        idx = indices[symbol]
        
        if idx < len(symbol_data):
            # Create bar
            row = symbol_data.iloc[idx]
            bar = Bar(
                symbol=symbol,
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            
            # Update index
            indices[symbol] = idx + 1
            
            # Publish BAR event to event bus
            if self.event_bus:
                from ..core.events.types import Event, EventType
                event = Event(
                    event_type=EventType.BAR.value,
                    payload={
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'bar': bar
                    },
                    source_id=self.handler_id
                )
                self.event_bus.publish(event)
            
            return True
        
        return False
    
    def has_more_data(self) -> bool:
        """Check if more data is available."""
        data_dict = self._get_active_data()
        indices = self._get_active_indices()
        
        return any(
            indices.get(symbol, 0) < len(data_dict.get(symbol, []))
            for symbol in self.symbols
        )
    
    def reset(self) -> None:
        """Reset to beginning of data."""
        # Reset indices
        for symbol in self.symbols:
            self.current_indices[symbol] = 0
        
        # Reset split indices
        if self.active_split and self.active_split in self.splits:
            split = self.splits[self.active_split]
            for symbol in split.indices:
                split.indices[symbol] = 0
        
        # Reset timeline
        self._timeline_idx = 0
    
    # Implements DataAccessor protocol
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get most recent bar for symbol."""
        data_dict = self._get_active_data()
        indices = self._get_active_indices()
        
        if symbol not in data_dict:
            return None
        
        idx = indices.get(symbol, 0)
        if idx == 0:
            return None
        
        # Get the last emitted bar
        row = data_dict[symbol].iloc[idx - 1]
        return Bar(
            symbol=symbol,
            timestamp=row.name,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Bar]:
        """Get last N bars for symbol."""
        data_dict = self._get_active_data()
        indices = self._get_active_indices()
        
        if symbol not in data_dict:
            return []
        
        idx = indices.get(symbol, 0)
        if idx == 0:
            return []
        
        # Get the last n emitted bars
        start_idx = max(0, idx - n)
        bars = []
        
        for i in range(start_idx, idx):
            row = data_dict[symbol].iloc[i]
            bar = Bar(
                symbol=symbol,
                timestamp=row.name,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            bars.append(bar)
        
        return bars
    
    def get_bar_at_index(self, symbol: str, index: int) -> Optional[Bar]:
        """Get bar at specific index."""
        data_dict = self._get_active_data()
        
        if symbol not in data_dict:
            return None
        
        data = data_dict[symbol]
        if index >= len(data):
            return None
        
        row = data.iloc[index]
        return Bar(
            symbol=symbol,
            timestamp=row.name,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
    
    # Implements DataSplitter protocol (can be moved to capability)
    def setup_split(self, method: str = 'ratio', **kwargs) -> None:
        """Set up train/test split."""
        if not self.data:
            raise ValueError("No data loaded")
        
        train_ratio = kwargs.get('train_ratio', 0.7)
        split_date = kwargs.get('split_date')
        
        for symbol, df in self.data.items():
            if method == 'ratio':
                split_idx = int(len(df) * train_ratio)
                train_data = df.iloc[:split_idx]
                test_data = df.iloc[split_idx:]
            elif method == 'date':
                if not split_date:
                    raise ValueError("split_date required for date-based split")
                split_date = pd.to_datetime(split_date)
                train_data = df[df.index < split_date]
                test_data = df[df.index >= split_date]
            else:
                raise ValueError(f"Unknown split method: {method}")
            
            # Create train split
            if "train" not in self.splits:
                self.splits["train"] = DataSplit(
                    name="train",
                    data={},
                    start_date=train_data.index[0],
                    end_date=train_data.index[-1]
                )
            self.splits["train"].data[symbol] = train_data
            self.splits["train"].indices[symbol] = 0
            
            # Create test split
            if "test" not in self.splits:
                self.splits["test"] = DataSplit(
                    name="test",
                    data={},
                    start_date=test_data.index[0],
                    end_date=test_data.index[-1]
                )
            self.splits["test"].data[symbol] = test_data
            self.splits["test"].indices[symbol] = 0
    
    def set_active_split(self, split_name: Optional[str]) -> None:
        """Set active data split."""
        if split_name and split_name not in self.splits:
            raise ValueError(f"Unknown split: {split_name}")
        
        self.active_split = split_name
        self.reset()
    
    def get_split_info(self) -> Dict[str, Any]:
        """Get information about current splits."""
        return {
            'active_split': self.active_split,
            'splits_configured': len(self.splits) > 0,
            'symbols': {
                symbol: {
                    'train_size': len(self.splits.get('train', DataSplit('', {}, datetime.now(), datetime.now())).data.get(symbol, [])),
                    'test_size': len(self.splits.get('test', DataSplit('', {}, datetime.now(), datetime.now())).data.get(symbol, [])),
                    'full_size': len(self.data.get(symbol, []))
                }
                for symbol in self.symbols
            }
        }
    
    # Implements HasLifecycle protocol (can be moved to capability)
    def start(self) -> None:
        """Start the data handler."""
        self._running = True
        if hasattr(self, 'log_info'):
            self.log_info("Data handler started")
    
    def stop(self) -> None:
        """Stop the data handler."""
        self._running = False
        if hasattr(self, 'log_info'):
            self.log_info("Data handler stopped")
    
    # Private helper methods - no inheritance complexity
    def _get_active_data(self) -> Dict[str, pd.DataFrame]:
        """Get the currently active dataset."""
        if self.active_split and self.active_split in self.splits:
            return self.splits[self.active_split].data
        return self.data
    
    def _get_active_indices(self) -> Dict[str, int]:
        """Get the currently active indices."""
        if self.active_split and self.active_split in self.splits:
            return self.splits[self.active_split].indices
        return self.current_indices
    
    def _build_timeline(self) -> None:
        """Build synchronized timeline across all symbols."""
        self._timeline = []
        
        for symbol, df in self.data.items():
            for timestamp in df.index:
                self._timeline.append((timestamp, symbol))
        
        # Sort by timestamp
        self._timeline.sort(key=lambda x: x[0])
    
    def _build_split_timeline(self, data_dict: Dict[str, pd.DataFrame]) -> List[Tuple[datetime, str]]:
        """Build timeline for a specific split."""
        timeline = []
        
        for symbol, df in data_dict.items():
            for timestamp in df.index:
                timeline.append((timestamp, symbol))
        
        timeline.sort(key=lambda x: x[0])
        return timeline


class StreamingDataHandler:
    """
    Simple streaming data handler - NO INHERITANCE!
    Shows how to implement streaming protocols.
    """
    
    def __init__(self, handler_id: str = "streaming_data"):
        self.handler_id = handler_id
        self.subscribed_symbols: List[str] = []
        self.latest_bars: Dict[str, Bar] = {}
        self._connected = False
    
    @property
    def name(self) -> str:
        return self.handler_id
    
    # Implements StreamingProvider protocol
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols."""
        self.subscribed_symbols.extend(symbols)
        # In real implementation, would connect to data feed
        
    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
    
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get latest bar from stream."""
        return self.latest_bars.get(symbol)
    
    # Implements HasLifecycle protocol
    def start(self) -> None:
        """Start streaming."""
        self._connected = True
        # In real implementation, would start data feed connection
    
    def stop(self) -> None:
        """Stop streaming."""
        self._connected = False
        # In real implementation, would close connections


class SimpleDataValidator:
    """
    Simple data validator - NO INHERITANCE!
    Implements DataValidator protocol.
    """
    
    def __init__(self):
        self.validation_rules = [
            'ohlc_relationships',
            'positive_volume',
            'no_duplicates',
            'chronological_order'
        ]
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data and return results."""
        errors = []
        warnings = []
        
        # Check required columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        
        if not errors:  # Only check relationships if we have the columns
            # Check OHLC relationships
            invalid_high = (data["high"] < data["low"]) | (data["high"] < data["open"]) | (data["high"] < data["close"])
            invalid_low = (data["low"] > data["open"]) | (data["low"] > data["close"])
            invalid_volume = data["volume"] < 0
            
            if invalid_high.any():
                errors.append(f"Invalid high prices in {invalid_high.sum()} bars")
            if invalid_low.any():
                errors.append(f"Invalid low prices in {invalid_low.sum()} bars")
            if invalid_volume.any():
                errors.append(f"Negative volume in {invalid_volume.sum()} bars")
            
            # Check for duplicates
            if data.index.duplicated().any():
                warnings.append(f"Found {data.index.duplicated().sum()} duplicate timestamps")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'metadata': {
                'total_bars': len(data),
                'date_range': (data.index[0], data.index[-1]) if len(data) > 0 else None
            }
        }
    
    def get_validation_rules(self) -> List[str]:
        """Get list of validation rule names."""
        return self.validation_rules.copy()


# Factory functions instead of inheritance
def create_data_handler(handler_type: str, **config) -> Any:
    """
    Factory function to create data handlers.
    No inheritance - just returns appropriate class.
    """
    handlers = {
        'historical': SimpleHistoricalDataHandler,
        'streaming': StreamingDataHandler
    }
    
    handler_class = handlers.get(handler_type)
    if not handler_class:
        raise ValueError(f"Unknown handler type: {handler_type}")
    
    return handler_class(**config)
