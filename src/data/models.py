"""
Data models for ADMF-PC - keeping the good parts from the original.

These are simple data classes with no inheritance.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

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
            "1M": 2592000
        }
        return mapping.get(self.value, 0)


@dataclass
class Bar:
    """
    Represents a single OHLCV bar - simple dataclass, no inheritance.
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
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
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
    def from_series(cls, series: pd.Series, symbol: str, timestamp: datetime = None) -> 'Bar':
        """Create Bar from pandas Series."""
        return cls(
            symbol=symbol,
            timestamp=timestamp or series.name or datetime.now(),
            open=series["open"],
            high=series["high"],
            low=series["low"],
            close=series["close"],
            volume=series["volume"]
        )


# Alias for backward compatibility
MarketData = Bar


@dataclass
class Tick:
    """Represents a single tick - simple dataclass, no inheritance."""
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
    Simple class, no inheritance.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ):
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
    Simple class with no inheritance.
    """
    
    def __init__(
        self,
        timestamps: Union[List[datetime], 'np.ndarray'],
        data: Dict[str, 'np.ndarray']
    ):
        if HAS_NUMPY:
            self.timestamps = np.array(timestamps)
        else:
            self.timestamps = list(timestamps)
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
        if not HAS_PANDAS:
            raise RuntimeError("pandas not available")
        
        timestamps = df.index.to_numpy() if HAS_NUMPY else df.index.tolist()
        data = {col: df[col].to_numpy() if HAS_NUMPY else df[col].tolist() 
                for col in df.columns}
        return cls(timestamps, data)
    
    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert to pandas DataFrame."""
        if not HAS_PANDAS:
            raise RuntimeError("pandas not available")
        
        if HAS_NUMPY and isinstance(self.timestamps, np.ndarray):
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
        return len(self.timestamps)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamps[idx],
            **{col: values[idx] for col, values in self.data.items()}
        }


@dataclass
class DataSplit:
    """Represents a train/test data split - simple dataclass."""
    name: str
    data: Dict[str, pd.DataFrame]  # symbol -> data
    start_date: datetime
    end_date: datetime
    indices: Dict[str, int] = field(default_factory=dict)  # symbol -> current index


@dataclass
class ValidationResult:
    """Result of data validation - simple dataclass."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
