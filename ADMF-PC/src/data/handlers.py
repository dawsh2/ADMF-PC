"""
Data handlers for ADMF-PC.

This module provides data loading and management for backtesting and
live trading within the containerized system.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from .models import Bar, Timeframe, DataView, TimeSeriesData
from ..core.components import Component, Lifecycle, EventCapable
from ..core.events import Event, EventType, create_market_event
from ..core.logging import StructuredLogger, ContainerLogger


@dataclass
class DataSplit:
    """Represents a train/test data split."""
    name: str
    data: Dict[str, pd.DataFrame]  # symbol -> data
    start_date: datetime
    end_date: datetime
    indices: Dict[str, int] = field(default_factory=dict)  # symbol -> current index


class DataHandler(Component, Lifecycle, EventCapable, ABC):
    """
    Abstract base class for data handlers.
    
    Data handlers are responsible for loading, managing, and emitting
    market data events within containers.
    """
    
    def __init__(self, handler_id: str = "data_handler"):
        """
        Initialize data handler.
        
        Args:
            handler_id: Unique identifier for this handler
        """
        self.handler_id = handler_id
        self._logger = StructuredLogger(f"DataHandler.{handler_id}")
        
        # Data storage
        self.symbols: List[str] = []
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_indices: Dict[str, int] = {}
        
        # Train/test splits
        self.splits: Dict[str, DataSplit] = {}
        self.active_split: Optional[str] = None
        
        # State
        self._initialized = False
        self._running = False
        
        # Event bus will be set during initialization
        self._event_bus = None
        self.container_id = None
    
    @property
    def component_id(self) -> str:
        """Component identifier."""
        return self.handler_id
    
    @property
    def event_bus(self):
        """Get event bus."""
        return self._event_bus
    
    @event_bus.setter
    def event_bus(self, value):
        """Set event bus."""
        self._event_bus = value
    
    # Abstract methods
    
    @abstractmethod
    def load_data(self, symbols: List[str]) -> None:
        """
        Load data for specified symbols.
        
        Args:
            symbols: List of symbols to load
        """
        pass
    
    @abstractmethod
    def update_bars(self) -> bool:
        """
        Update to next bar and emit event.
        
        Returns:
            True if more bars available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """
        Get the latest bar for a symbol.
        
        Args:
            symbol: Symbol to get bar for
            
        Returns:
            Latest bar or None
        """
        pass
    
    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Bar]:
        """
        Get the latest N bars for a symbol.
        
        Args:
            symbol: Symbol to get bars for
            n: Number of bars to retrieve
            
        Returns:
            List of bars (most recent last)
        """
        pass
    
    # Lifecycle methods
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the data handler."""
        self.event_bus = context.get('event_bus')
        self.container_id = context.get('container_id')
        
        if self.container_id:
            self._logger = ContainerLogger(
                "DataHandler",
                self.container_id,
                self.handler_id
            )
        
        self._initialized = True
        self._logger.info("Data handler initialized")
    
    def start(self) -> None:
        """Start the data handler."""
        if not self._initialized:
            raise RuntimeError("Data handler not initialized")
        
        self._running = True
        self._logger.info("Data handler started")
    
    def stop(self) -> None:
        """Stop the data handler."""
        self._running = False
        self._logger.info("Data handler stopped")
    
    def reset(self) -> None:
        """Reset the data handler."""
        # Reset indices
        for symbol in self.symbols:
            self.current_indices[symbol] = 0
        
        # Reset active split indices
        if self.active_split and self.active_split in self.splits:
            split = self.splits[self.active_split]
            for symbol in split.indices:
                split.indices[symbol] = 0
        
        self._logger.info("Data handler reset")
    
    def teardown(self) -> None:
        """Clean up resources."""
        self.data.clear()
        self.splits.clear()
        self.symbols.clear()
        self._initialized = False
        self._logger.info("Data handler torn down")
    
    # Event methods
    
    def initialize_events(self) -> None:
        """Initialize event subscriptions."""
        # Data handlers typically don't subscribe to events
        pass
    
    def teardown_events(self) -> None:
        """Clean up event subscriptions."""
        pass
    
    # Split management
    
    def setup_train_test_split(
        self,
        method: str = "ratio",
        train_ratio: float = 0.7,
        split_date: Optional[datetime] = None
    ) -> None:
        """
        Set up train/test data splits.
        
        Args:
            method: Split method ("ratio" or "date")
            train_ratio: Ratio of data for training (if method="ratio")
            split_date: Date to split at (if method="date")
        """
        if not self.data:
            raise ValueError("No data loaded")
        
        self._logger.info(
            f"Setting up train/test split",
            method=method,
            train_ratio=train_ratio,
            split_date=split_date
        )
        
        for symbol, df in self.data.items():
            if method == "ratio":
                split_idx = int(len(df) * train_ratio)
                train_data = df.iloc[:split_idx]
                test_data = df.iloc[split_idx:]
            
            elif method == "date":
                if not split_date:
                    raise ValueError("split_date required for date-based split")
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
        
        self._logger.info(
            f"Train/test split complete",
            train_size=len(self.splits["train"].data[self.symbols[0]]),
            test_size=len(self.splits["test"].data[self.symbols[0]])
        )
    
    def set_active_split(self, split_name: Optional[str]) -> None:
        """
        Set the active data split.
        
        Args:
            split_name: "train", "test", or None for full data
        """
        if split_name and split_name not in self.splits:
            raise ValueError(f"Unknown split: {split_name}")
        
        self.active_split = split_name
        self.reset()
        
        self._logger.info(f"Active split set to: {split_name or 'full'}")


class HistoricalDataHandler(DataHandler):
    """
    Data handler for historical backtesting data.
    
    This handler loads data from CSV files and emits bars in chronological
    order across all symbols.
    """
    
    def __init__(
        self,
        handler_id: str = "historical_data",
        data_dir: str = "data",
        timeframe: Timeframe = Timeframe.D1
    ):
        """
        Initialize historical data handler.
        
        Args:
            handler_id: Handler identifier
            data_dir: Directory containing CSV files
            timeframe: Data timeframe
        """
        super().__init__(handler_id)
        self.data_dir = Path(data_dir)
        self.timeframe = timeframe
        
        # Multi-symbol synchronization
        self._timeline: List[Tuple[datetime, str]] = []
        self._timeline_idx = 0
    
    def load_data(self, symbols: List[str]) -> None:
        """Load data from CSV files."""
        self.symbols = symbols
        self._logger.info(f"Loading data for symbols: {symbols}")
        
        for symbol in symbols:
            # Try to find CSV file
            csv_path = self.data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                # Try with lowercase
                csv_path = self.data_dir / f"{symbol.lower()}.csv"
            
            if not csv_path.exists():
                raise FileNotFoundError(f"Data file not found for {symbol}")
            
            # Load CSV
            df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Validate required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = set(required) - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns for {symbol}: {missing}")
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Store data
            self.data[symbol] = df
            self.current_indices[symbol] = 0
            
            self._logger.info(
                f"Loaded {symbol}",
                rows=len(df),
                start=df.index[0],
                end=df.index[-1]
            )
        
        # Build synchronized timeline
        self._build_timeline()
    
    def update_bars(self) -> bool:
        """Emit next bar across all symbols."""
        if not self._running:
            return False
        
        # Get active dataset
        data_dict = self._get_active_data()
        indices = self._get_active_indices()
        
        if self.active_split:
            # Use timeline for current split
            timeline = self._build_split_timeline(data_dict)
            if self._timeline_idx >= len(timeline):
                return False
            
            timestamp, symbol = timeline[self._timeline_idx]
            self._timeline_idx += 1
        else:
            # Use main timeline
            if self._timeline_idx >= len(self._timeline):
                return False
            
            timestamp, symbol = self._timeline[self._timeline_idx]
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
                volume=row['volume'],
                timeframe=self.timeframe
            )
            
            # Update index
            indices[symbol] = idx + 1
            
            # Emit event
            if self.event_bus:
                event = create_market_event(
                    EventType.BAR,
                    symbol=symbol,
                    timestamp=timestamp,
                    data=bar.to_dict(),
                    source_id=self.handler_id,
                    container_id=self.container_id
                )
                self.event_bus.publish(event)
            
            return True
        
        return False
    
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get the latest bar for a symbol."""
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
            volume=row['volume'],
            timeframe=self.timeframe
        )
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Bar]:
        """Get the latest N bars for a symbol."""
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
                volume=row['volume'],
                timeframe=self.timeframe
            )
            bars.append(bar)
        
        return bars
    
    def reset(self) -> None:
        """Reset handler state."""
        super().reset()
        self._timeline_idx = 0
    
    # Private methods
    
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
        
        self._logger.info(f"Built timeline with {len(self._timeline)} events")
    
    def _build_split_timeline(self, data_dict: Dict[str, pd.DataFrame]) -> List[Tuple[datetime, str]]:
        """Build timeline for a specific split."""
        timeline = []
        
        for symbol, df in data_dict.items():
            for timestamp in df.index:
                timeline.append((timestamp, symbol))
        
        timeline.sort(key=lambda x: x[0])
        return timeline