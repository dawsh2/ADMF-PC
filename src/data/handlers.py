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
        
        # Event bus and container - set by container
        self.event_bus = None
        self.container = None
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return self.handler_id
    
    def set_event_bus(self, event_bus) -> None:
        """Set the event bus for publishing events."""
        self.event_bus = event_bus
    
    def set_container(self, container) -> None:
        """Set the container reference for parent publishing."""
        self.container = container
    
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
        logger.info(f"Data handler execute() called for symbols: {self.symbols}")
        
        if not self._running:
            self.start()
        
        # Stream all bars
        bars_streamed = 0
        max_bars = getattr(self, 'max_bars', float('inf'))
        
        # Handle None max_bars
        if max_bars is None:
            max_bars = float('inf')
        
        logger.info(f"Starting to stream bars, max_bars: {max_bars}, has_data: {self.has_more_data()}")
        
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
            
            # Calculate original dataset bar index for consistent sparse storage
            original_bar_index = self._get_original_bar_index(symbol, idx)
            
            # Update index
            indices[symbol] = idx + 1
            
            # Publish BAR event to shared root event bus
            logger.debug(f"Publishing BAR event - container: {self.container}, event_bus: {self.event_bus}")
            if self.container:
                from ..core.events.types import Event, EventType
                event = Event(
                    event_type=EventType.BAR.value,
                    payload={
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'bar': bar,
                        'original_bar_index': original_bar_index,  # For consistent sparse storage
                        'split_bar_index': idx + 1  # Current position within split
                    },
                    source_id=f"data_{symbol}",
                    container_id=self.container.container_id if self.container else None
                )
                logger.info(f"📊 Publishing BAR event #{self._timeline_idx} for {symbol} at {timestamp}")
                # Publish directly to container's event bus (should be shared root bus)
                self.container.event_bus.publish(event)
            elif self.event_bus:
                # Fallback to local bus if no container
                from ..core.events.types import Event, EventType
                event = Event(
                    event_type=EventType.BAR.value,
                    payload={
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'bar': bar,
                        'original_bar_index': original_bar_index,  # For consistent sparse storage
                        'split_bar_index': idx + 1  # Current position within split
                    },
                    source_id=self.handler_id
                )
                logger.info(f"📊 Publishing BAR event #{self._timeline_idx} for {symbol} at {timestamp}")
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
    
    def setup_wfv_window(self, window_num: int, total_windows: int, 
                         phase: str, dataset_split: str = 'train') -> None:
        """
        Setup walk-forward validation window with sliding window methodology.
        
        Uses sliding windows where each window's test data becomes part of 
        the next window's training data.
        
        Args:
            window_num: Current window number (1-based)
            total_windows: Total number of WFV windows
            phase: 'train' or 'test' phase
            dataset_split: 'train' or 'test' split of full dataset
        """
        if window_num < 1 or window_num > total_windows:
            raise ValueError(f"Window {window_num} must be between 1 and {total_windows}")
        
        # Get the base dataset (train or test split)
        base_data = self._get_base_dataset(dataset_split)
        
        if not base_data:
            raise ValueError(f"No data available for {dataset_split} split")
        
        # Calculate sliding window boundaries
        window_data = self._calculate_wfv_window(base_data, window_num, total_windows, phase)
        
        # Create WFV split
        split_name = f"wfv_w{window_num}_{phase}"
        self.splits[split_name] = DataSplit(
            name=split_name,
            data=window_data,
            start_date=min(df.index[0] for df in window_data.values() if len(df) > 0),
            end_date=max(df.index[-1] for df in window_data.values() if len(df) > 0)
        )
        
        # Initialize indices for the split
        for symbol in window_data:
            self.splits[split_name].indices[symbol] = 0
        
        # Set as active split
        self.set_active_split(split_name)
        
        logger.info(f"Setup WFV window {window_num}/{total_windows} ({phase}) on {dataset_split} split")
        logger.info(f"Window data: {sum(len(df) for df in window_data.values())} bars total")
    
    def _get_base_dataset(self, dataset_split: str) -> Dict[str, pd.DataFrame]:
        """
        Get the base dataset for WFV (train or test split).
        
        Args:
            dataset_split: 'train', 'test', or 'full'
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        if dataset_split == 'train' and 'train' in self.splits:
            return self.splits['train'].data
        elif dataset_split == 'test' and 'test' in self.splits:
            return self.splits['test'].data
        elif dataset_split == 'full':
            return self.data
        else:
            logger.warning(f"Split '{dataset_split}' not found, using full dataset")
            return self.data
    
    def _calculate_wfv_window(self, base_data: Dict[str, pd.DataFrame], 
                             window_num: int, total_windows: int, 
                             phase: str) -> Dict[str, pd.DataFrame]:
        """
        Calculate sliding window boundaries for WFV.
        
        Uses expanding training windows with fixed test sizes for proper
        walk-forward validation methodology.
        
        Args:
            base_data: Base dataset dictionary
            window_num: Current window number (1-based)
            total_windows: Total number of windows
            phase: 'train' or 'test'
            
        Returns:
            Window data dictionary
        """
        window_data = {}
        
        for symbol, df in base_data.items():
            if len(df) == 0:
                window_data[symbol] = df.copy()
                continue
            
            total_bars = len(df)
            
            # Calculate window sizing
            # Reserve some data for final test window
            usable_bars = int(total_bars * 0.9)  # Use 90% for WFV, reserve 10% for final
            test_size = max(50, usable_bars // (total_windows * 4))  # Test size: ~1/4 of average window
            
            # Calculate window boundaries (sliding window approach)
            step_size = max(test_size // 2, 25)  # Overlap windows by 50%
            
            # Window start positions
            window_start = (window_num - 1) * step_size
            
            if phase == 'train':
                # Training: expanding window from start to current window end
                train_end = window_start + (window_num * test_size) + (window_num * step_size)
                train_end = min(train_end, usable_bars - test_size)  # Leave room for test
                
                start_idx = 0
                end_idx = train_end
                
            else:  # phase == 'test'
                # Testing: fixed size window after training data
                train_end = window_start + (window_num * test_size) + (window_num * step_size)
                train_end = min(train_end, usable_bars - test_size)
                
                start_idx = train_end
                end_idx = min(train_end + test_size, usable_bars)
            
            # Ensure valid boundaries
            start_idx = max(0, start_idx)
            end_idx = min(end_idx, total_bars)
            
            if start_idx >= end_idx:
                logger.warning(f"Invalid window boundaries for {symbol} window {window_num}: "
                              f"start={start_idx}, end={end_idx}")
                window_data[symbol] = df.iloc[0:0].copy()  # Empty DataFrame
            else:
                window_data[symbol] = df.iloc[start_idx:end_idx].copy()
                
                logger.debug(f"WFV Window {window_num} ({phase}) for {symbol}: "
                            f"bars {start_idx}-{end_idx} ({end_idx-start_idx} bars)")
        
        return window_data
    
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
    
    def _get_original_bar_index(self, symbol: str, split_index: int) -> int:
        """
        Get the original dataset bar index for a given split index.
        
        Args:
            symbol: Trading symbol
            split_index: Index within the current split (0-based)
            
        Returns:
            Original dataset bar index (0-based)
        """
        if not self.active_split or self.active_split not in self.splits:
            # No split active, use direct index
            return split_index
        
        # Get the split data and original data
        split_data = self.splits[self.active_split].data.get(symbol)
        original_data = self.data.get(symbol)
        
        if split_data is None or original_data is None:
            return split_index
        
        if split_index >= len(split_data):
            return split_index
        
        # Get the timestamp at the split index
        split_timestamp = split_data.index[split_index]
        
        # Find this timestamp in the original dataset
        try:
            # Use get_loc to find the position of this timestamp in original data
            original_index = original_data.index.get_loc(split_timestamp)
            return original_index
        except KeyError:
            # Timestamp not found (shouldn't happen), fall back to split index
            logger.warning(f"Timestamp {split_timestamp} not found in original dataset for {symbol}")
            return split_index


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
