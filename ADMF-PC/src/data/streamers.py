"""
Data streaming components implementing BACKTEST.MD patterns.

Provides:
- HistoricalDataStreamer: For full backtest pattern
- SignalLogStreamer: For signal replay pattern
"""

from typing import Dict, List, Any, Optional, AsyncIterator, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import logging
from dataclasses import dataclass

from ..core.events import Event, EventType
from .models import Bar
from .loaders import CSVLoader

logger = logging.getLogger(__name__)


@dataclass
class StreamedBar:
    """A bar of market data being streamed."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    

@dataclass
class StreamedSignal:
    """A signal being streamed from logs."""
    timestamp: datetime
    strategy_id: str
    symbol: str
    direction: str  # BUY, SELL, HOLD
    strength: float
    metadata: Dict[str, Any]


class HistoricalDataStreamer:
    """
    Streams historical market data for backtesting.
    
    Part of the Full Backtest pattern from BACKTEST.MD.
    Emits BAR events to the IndicatorHub.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize historical data streamer.
        
        Args:
            config: Data configuration with:
                - file_path: Path to data file(s)
                - symbols: List of symbols to stream
                - start_date: Optional start date
                - end_date: Optional end date
                - frequency: Bar frequency (1min, 5min, etc)
        """
        self.config = config
        self.data_loader = CSVLoader(data_dir=Path(config.get('file_path', 'data')).parent)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._initialized = False
        
    async def initialize(self) -> None:
        """Load and prepare data for streaming."""
        symbols = self.config.get('symbols', [])
        
        for symbol in symbols:
            try:
                # Load data using CSV loader
                df = self.data_loader.load(symbol)
                
                # Apply date filters
                start_date = self.config.get('start_date')
                end_date = self.config.get('end_date')
                
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]
                
                # Apply max_bars limit
                max_bars = self.config.get('max_bars')
                if max_bars and len(df) > max_bars:
                    df = df.iloc[:max_bars]
                
                self._data_cache[symbol] = df
                logger.info(f"Loaded {len(df)} bars for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
                
        self._initialized = True
        
    async def stream_bars(self) -> AsyncIterator[Tuple[datetime, Dict[str, StreamedBar]]]:
        """
        Stream market data bars chronologically.
        
        Yields:
            Tuple of (timestamp, bars) where bars is a dict of symbol -> StreamedBar
        """
        if not self._initialized:
            await self.initialize()
            
        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for df in self._data_cache.values():
            all_timestamps.update(df.index)
            
        # Sort chronologically
        sorted_timestamps = sorted(all_timestamps)
        
        # Stream bars for each timestamp
        for timestamp in sorted_timestamps:
            bars = {}
            
            # Collect bars for all symbols at this timestamp
            for symbol, df in self._data_cache.items():
                if timestamp in df.index:
                    row = df.loc[timestamp]
                    bars[symbol] = StreamedBar(
                        timestamp=timestamp,
                        symbol=symbol,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
            
            if bars:
                yield timestamp, bars
                
    def to_event(self, timestamp: datetime, bars: Dict[str, StreamedBar]) -> Event:
        """Convert streamed bars to a BAR event."""
        return Event(
            event_type=EventType.BAR,
            source_id="data_streamer",
            timestamp=timestamp,
            payload={
                'timestamp': timestamp,
                'bars': {
                    symbol: {
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    }
                    for symbol, bar in bars.items()
                }
            }
        )


class SignalLogStreamer:
    """
    Streams saved signals for signal replay pattern.
    
    Part of the Signal Replay pattern from BACKTEST.MD.
    Used for ensemble optimization without recomputing indicators.
    """
    
    def __init__(self, signal_log_path: str):
        """
        Initialize signal log streamer.
        
        Args:
            signal_log_path: Path to signal log file
        """
        self.signal_log_path = Path(signal_log_path)
        self._signals: List[StreamedSignal] = []
        self._initialized = False
        
    async def initialize(self) -> None:
        """Load signals from log file."""
        if not self.signal_log_path.exists():
            raise FileNotFoundError(f"Signal log not found: {self.signal_log_path}")
            
        with open(self.signal_log_path, 'r') as f:
            signal_data = json.load(f)
            
        # Parse signals
        for signal_dict in signal_data.get('signals', []):
            signal = StreamedSignal(
                timestamp=pd.to_datetime(signal_dict['timestamp']),
                strategy_id=signal_dict['strategy_id'],
                symbol=signal_dict['symbol'],
                direction=signal_dict['direction'],
                strength=signal_dict['strength'],
                metadata=signal_dict.get('metadata', {})
            )
            self._signals.append(signal)
            
        # Sort by timestamp
        self._signals.sort(key=lambda s: s.timestamp)
        
        logger.info(f"Loaded {len(self._signals)} signals from log")
        self._initialized = True
        
    async def stream_signals(self) -> AsyncIterator[Tuple[datetime, List[StreamedSignal]]]:
        """
        Stream signals chronologically.
        
        Yields:
            Tuple of (timestamp, signals) where signals is a list of signals at that time
        """
        if not self._initialized:
            await self.initialize()
            
        # Group signals by timestamp
        from itertools import groupby
        
        for timestamp, signal_group in groupby(self._signals, key=lambda s: s.timestamp):
            signals = list(signal_group)
            yield timestamp, signals
            
    def to_event(self, timestamp: datetime, signals: List[StreamedSignal]) -> Event:
        """Convert streamed signals to a SIGNAL event."""
        return Event(
            event_type=EventType.SIGNAL,
            source_id="signal_streamer",
            timestamp=timestamp,
            payload={
                'timestamp': timestamp,
                'signals': [
                    {
                        'strategy_id': signal.strategy_id,
                        'symbol': signal.symbol,
                        'direction': signal.direction,
                        'strength': signal.strength,
                        'metadata': signal.metadata
                    }
                    for signal in signals
                ]
            }
        )


class RealtimeDataStreamer:
    """
    Streams real-time market data for live trading.
    
    This is a placeholder for a real implementation that would:
    - Connect to market data feeds
    - Handle reconnection and failover
    - Provide data quality checks
    - Support multiple data providers
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize realtime streamer."""
        self.config = config
        logger.warning("RealtimeDataStreamer is not yet implemented")
        
    async def stream_bars(self) -> AsyncIterator[Tuple[datetime, Dict[str, StreamedBar]]]:
        """Stream live market data."""
        raise NotImplementedError("Realtime streaming not yet implemented")