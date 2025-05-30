"""
Data models for ADMF-PC.

This module defines the core data structures used throughout the system
for representing market data.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from enum import Enum


class Timeframe(Enum):
    """Standard timeframes for market data."""
    TICK = "tick"
    S1 = "1s"
    S5 = "5s"
    S15 = "15s"
    S30 = "30s"
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"
    
    @property
    def seconds(self) -> int:
        """Get timeframe duration in seconds."""
        mapping = {
            "tick": 0,
            "1s": 1,
            "5s": 5,
            "15s": 15,
            "30s": 30,
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800,
            "1M": 2592000  # Approximate
        }
        return mapping.get(self.value, 0)


@dataclass
class Bar:
    """
    Represents a single OHLCV bar of market data.
    
    This is the fundamental data structure for price data in the system.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Optional[Timeframe] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate bar data."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than Low ({self.low})")
        
        if self.high < self.open or self.high < self.close:
            raise ValueError("High must be >= Open and Close")
        
        if self.low > self.open or self.low > self.close:
            raise ValueError("Low must be <= Open and Close")
        
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
    
    @property
    def range(self) -> float:
        """Price range of the bar."""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """Body size of the bar."""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """Whether this is a bullish (green) bar."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Whether this is a bearish (red) bar."""
        return self.close < self.open
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "timeframe": self.timeframe.value if self.timeframe else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bar':
        """Create Bar from dictionary."""
        # Handle timestamp
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # Handle timeframe
        timeframe = None
        if "timeframe" in data and data["timeframe"]:
            timeframe = Timeframe(data["timeframe"])
        
        return cls(
            symbol=data["symbol"],
            timestamp=timestamp,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            timeframe=timeframe,
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_series(cls, series: pd.Series, symbol: str) -> 'Bar':
        """Create Bar from pandas Series."""
        return cls(
            symbol=symbol,
            timestamp=series.name if isinstance(series.name, datetime) else datetime.now(),
            open=series["open"],
            high=series["high"],
            low=series["low"],
            close=series["close"],
            volume=series["volume"]
        )


@dataclass
class Tick:
    """Represents a single tick (trade) in the market."""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread if both are available."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


class DataView:
    """
    Read-only view of data for memory-efficient access.
    
    This provides a window into a larger dataset without copying.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ):
        """
        Initialize data view.
        
        Args:
            data: Underlying DataFrame
            start_idx: Start index for view
            end_idx: End index for view (None for end of data)
        """
        self._data = data
        self._start_idx = start_idx
        self._end_idx = end_idx or len(data)
        self._current_idx = start_idx
    
    def get_current(self) -> Optional[pd.Series]:
        """Get current data point."""
        if self._current_idx < self._end_idx:
            return self._data.iloc[self._current_idx]
        return None
    
    def advance(self) -> bool:
        """Move to next data point."""
        if self._current_idx < self._end_idx - 1:
            self._current_idx += 1
            return True
        return False
    
    def get_window(self, size: int) -> pd.DataFrame:
        """Get a window of data points."""
        start = max(0, self._current_idx - size + 1)
        end = self._current_idx + 1
        return self._data.iloc[start:end]
    
    def reset(self) -> None:
        """Reset to start of view."""
        self._current_idx = self._start_idx
    
    @property
    def has_data(self) -> bool:
        """Check if more data is available."""
        return self._current_idx < self._end_idx - 1
    
    @property
    def progress(self) -> float:
        """Get progress through the view (0-1)."""
        total = self._end_idx - self._start_idx
        if total == 0:
            return 1.0
        progress = (self._current_idx - self._start_idx + 1) / total
        return min(1.0, progress)


class TimeSeriesData:
    """
    Efficient storage for time series data.
    
    Stores timestamps and values separately for better memory usage.
    """
    
    def __init__(
        self,
        timestamps: Union[List[datetime], np.ndarray],
        data: Dict[str, np.ndarray]
    ):
        """
        Initialize time series data.
        
        Args:
            timestamps: Array of timestamps
            data: Dictionary of column name to value arrays
        """
        self.timestamps = np.array(timestamps)
        self.data = data
        self._validate()
    
    def _validate(self):
        """Validate data consistency."""
        n_timestamps = len(self.timestamps)
        for column, values in self.data.items():
            if len(values) != n_timestamps:
                raise ValueError(
                    f"Column '{column}' has {len(values)} values, "
                    f"expected {n_timestamps}"
                )
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'TimeSeriesData':
        """Create from pandas DataFrame."""
        timestamps = df.index.to_numpy()
        data = {col: df[col].to_numpy() for col in df.columns}
        return cls(timestamps, data)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        # Convert timestamps back to DatetimeIndex
        if isinstance(self.timestamps, np.ndarray) and len(self.timestamps) > 0:
            # If timestamps are datetime objects in numpy array
            index = pd.DatetimeIndex(self.timestamps)
        else:
            index = self.timestamps
        return pd.DataFrame(self.data, index=index)
    
    def get_view(self, start_idx: int = 0, end_idx: Optional[int] = None) -> 'TimeSeriesData':
        """Get a view of the data."""
        end_idx = end_idx or len(self.timestamps)
        
        view_timestamps = self.timestamps[start_idx:end_idx]
        view_data = {
            col: values[start_idx:end_idx]
            for col, values in self.data.items()
        }
        
        return TimeSeriesData(view_timestamps, view_data)
    
    def __len__(self) -> int:
        """Number of data points."""
        return len(self.timestamps)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get data point at index."""
        return {
            "timestamp": self.timestamps[idx],
            **{col: values[idx] for col, values in self.data.items()}
        }


@dataclass
class MarketData:
    """
    Container for market data used throughout the system.
    
    This is a simple wrapper that can hold either DataFrame or TimeSeriesData.
    """
    symbol: str
    timeframe: Union[str, Timeframe]
    data: Union['pd.DataFrame', TimeSeriesData, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_dataframe(self) -> bool:
        """Check if data is a DataFrame."""
        return HAS_PANDAS and isinstance(self.data, pd.DataFrame)
    
    @property
    def is_timeseries(self) -> bool:
        """Check if data is TimeSeriesData."""
        return isinstance(self.data, TimeSeriesData)
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to DataFrame if possible."""
        if self.is_dataframe:
            return self.data
        elif self.is_timeseries:
            return self.data.to_dataframe()
        else:
            # Assume dict format
            if HAS_PANDAS:
                return pd.DataFrame(self.data)
            else:
                raise RuntimeError("pandas not available")
    
    def get_bars(self, start_idx: int = 0, end_idx: Optional[int] = None) -> List[Bar]:
        """Get bars from the data."""
        if self.is_dataframe:
            df = self.data.iloc[start_idx:end_idx]
            bars = []
            for idx, row in df.iterrows():
                bars.append(Bar(
                    timestamp=idx,
                    open=row.get('open', 0),
                    high=row.get('high', 0),
                    low=row.get('low', 0),
                    close=row.get('close', 0),
                    volume=row.get('volume', 0)
                ))
            return bars
        else:
            # Handle other formats
            return []