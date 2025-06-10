"""
Data streaming components using Protocol+Composition - NO INHERITANCE!

Simple streaming classes that implement protocols through duck typing.
"""

from typing import Dict, List, Any, Optional, AsyncIterator, Tuple, Iterator
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import asyncio
from dataclasses import dataclass

from ..models import Bar, Timeframe  
from ..loaders import SimpleCSVLoader


@dataclass
class StreamedBar:
    """A bar of market data being streamed - simple dataclass."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_bar(self) -> Bar:
        """Convert to Bar object."""
        return Bar(
            symbol=self.symbol,
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume
        )


@dataclass
class StreamedSignal:
    """A signal being streamed from logs - simple dataclass."""
    timestamp: datetime
    strategy_id: str
    symbol: str
    direction: str  # BUY, SELL, HOLD
    strength: float
    metadata: Dict[str, Any]


class SimpleHistoricalStreamer:
    """
    Simple historical data streamer - NO INHERITANCE!
    
    Implements streaming protocols through duck typing.
    Enhanced through capabilities, not inheritance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config.get('data_dir', 'data')
        self.symbols = config.get('symbols', [])
        
        # Data storage
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._initialized = False
        
        # Create loader
        self.loader = SimpleCSVLoader(self.data_dir)
    
    @property
    def name(self) -> str:
        return "historical_streamer"
    
    # Implements DataProvider protocol  
    async def load_data(self, symbols: List[str]) -> bool:
        """Load data for streaming."""
        self.symbols = symbols
        
        try:
            for symbol in symbols:
                # Load data
                df = self.loader.load(symbol)
                
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
                
                if hasattr(self, 'log_info'):  # If logging capability added
                    self.log_info(f"Loaded {len(df)} bars for {symbol}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            if hasattr(self, 'log_error'):
                self.log_error(f"Failed to load data: {e}")
            return False
    
    def get_symbols(self) -> List[str]:
        """Get loaded symbols."""
        return self.symbols.copy()
    
    def has_data(self, symbol: str) -> bool:
        """Check if data exists for symbol."""
        return symbol in self._data_cache
    
    # Implements streaming functionality
    async def stream_bars(self) -> AsyncIterator[Tuple[datetime, Dict[str, StreamedBar]]]:
        """
        Stream market data bars chronologically.
        
        Yields:
            Tuple of (timestamp, bars) where bars is a dict of symbol -> StreamedBar
        """
        if not self._initialized:
            await self.load_data(self.symbols)
        
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
                
                # Emit event if capability available
                if hasattr(self, 'emit_event'):
                    self.emit_event('bars_streamed', {
                        'timestamp': timestamp,
                        'symbol_count': len(bars)
                    })
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.001)
    
    def to_event_dict(self, timestamp: datetime, bars: Dict[str, StreamedBar]) -> Dict[str, Any]:
        """Convert streamed bars to event payload."""
        return {
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


class SimpleSignalStreamer:
    """
    Simple signal streamer for replay - NO INHERITANCE!
    
    Streams saved signals for ensemble optimization.
    """
    
    def __init__(self, signal_log_path: str):
        self.signal_log_path = Path(signal_log_path)
        self._signals: List[StreamedSignal] = []
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "signal_streamer"
    
    async def load_signals(self) -> bool:
        """Load signals from log file."""
        try:
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
            
            if hasattr(self, 'log_info'):
                self.log_info(f"Loaded {len(self._signals)} signals from log")
            
            self._initialized = True
            return True
            
        except Exception as e:
            if hasattr(self, 'log_error'):
                self.log_error(f"Failed to load signals: {e}")
            return False
    
    async def stream_signals(self) -> AsyncIterator[Tuple[datetime, List[StreamedSignal]]]:
        """
        Stream signals chronologically.
        
        Yields:
            Tuple of (timestamp, signals) where signals is a list of signals at that time
        """
        if not self._initialized:
            await self.load_signals()
        
        # Group signals by timestamp
        from itertools import groupby
        
        for timestamp, signal_group in groupby(self._signals, key=lambda s: s.timestamp):
            signals = list(signal_group)
            yield timestamp, signals
            
            # Emit event if capability available
            if hasattr(self, 'emit_event'):
                self.emit_event('signals_streamed', {
                    'timestamp': timestamp,
                    'signal_count': len(signals)
                })
    
    def to_event_dict(self, timestamp: datetime, signals: List[StreamedSignal]) -> Dict[str, Any]:
        """Convert streamed signals to event payload."""
        return {
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


class SimpleRealTimeStreamer:
    """
    Simple real-time data streamer - NO INHERITANCE!
    
    Placeholder for actual real-time implementation.
    Shows how to implement streaming protocols.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.websocket_url = config.get('websocket_url')
        self.api_key = config.get('api_key')
        
        # State
        self.subscribed_symbols: List[str] = []
        self.latest_bars: Dict[str, Bar] = {}
        self._connected = False
    
    @property
    def name(self) -> str:
        return "realtime_streamer"
    
    # Implements StreamingProvider protocol
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols."""
        self.subscribed_symbols.extend(symbols)
        
        if hasattr(self, 'log_info'):
            self.log_info(f"Subscribed to symbols: {symbols}")
        
        # In real implementation, would send subscription message to websocket
    
    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
        
        if hasattr(self, 'log_info'):
            self.log_info(f"Unsubscribed from symbols: {symbols}")
    
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get latest bar from stream."""
        return self.latest_bars.get(symbol)
    
    # Implements HasLifecycle protocol
    def start(self) -> None:
        """Start real-time streaming."""
        self._connected = True
        
        if hasattr(self, 'log_info'):
            self.log_info("Real-time streaming started")
        
        # In real implementation, would establish websocket connection
    
    def stop(self) -> None:
        """Stop real-time streaming."""
        self._connected = False
        
        if hasattr(self, 'log_info'):
            self.log_info("Real-time streaming stopped")
        
        # In real implementation, would close websocket connection
    
    def reset(self) -> None:
        """Reset streamer state."""
        self.latest_bars.clear()
        self.subscribed_symbols.clear()
        self._connected = False
    
    async def process_message(self, message: Dict[str, Any]) -> None:
        """Process incoming real-time message."""
        # Placeholder for message processing
        # In real implementation, would parse market data messages
        
        if 'symbol' in message and 'price_data' in message:
            symbol = message['symbol']
            price_data = message['price_data']
            
            # Create bar from message
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.now(),
                open=price_data.get('open', 0),
                high=price_data.get('high', 0),
                low=price_data.get('low', 0),
                close=price_data.get('close', 0),
                volume=price_data.get('volume', 0)
            )
            
            # Update latest bar
            self.latest_bars[symbol] = bar
            
            # Emit event if capability available
            if hasattr(self, 'emit_event'):
                self.emit_event('real_time_bar', {
                    'symbol': symbol,
                    'bar': bar.to_dict()
                })


class MultiSourceStreamer:
    """
    Combines multiple streamers - NO INHERITANCE!
    
    Shows composition pattern instead of inheritance.
    """
    
    def __init__(self, streamers: List[Any]):
        self.streamers = streamers
        self._active_streamers: List[Any] = []
    
    @property
    def name(self) -> str:
        return "multi_source_streamer"
    
    def add_streamer(self, streamer: Any) -> None:
        """Add a streamer to the collection."""
        self.streamers.append(streamer)
    
    def start_all(self) -> None:
        """Start all streamers that support lifecycle."""
        for streamer in self.streamers:
            if hasattr(streamer, 'start'):
                streamer.start()
                self._active_streamers.append(streamer)
    
    def stop_all(self) -> None:
        """Stop all active streamers."""
        for streamer in self._active_streamers:
            if hasattr(streamer, 'stop'):
                streamer.stop()
        self._active_streamers.clear()
    
    async def stream_combined(self) -> AsyncIterator[Tuple[datetime, Any, Dict[str, Any]]]:
        """
        Stream data from all sources combined.
        
        Yields:
            Tuple of (timestamp, source_streamer, data)
        """
        # This would coordinate multiple async streamers
        # For now, just a placeholder
        for streamer in self._active_streamers:
            if hasattr(streamer, 'stream_bars'):
                async for timestamp, data in streamer.stream_bars():
                    yield timestamp, streamer, data


# Factory functions instead of inheritance
def create_streamer(streamer_type: str, **config) -> Any:
    """
    Factory function to create streamers.
    No inheritance - just returns appropriate class.
    """
    streamers = {
        'historical': SimpleHistoricalStreamer,
        'signal': SimpleSignalStreamer,
        'realtime': SimpleRealTimeStreamer,
        'multi': MultiSourceStreamer
    }
    
    streamer_class = streamers.get(streamer_type)
    if not streamer_class:
        raise ValueError(f"Unknown streamer type: {streamer_type}")
    
    if streamer_type == 'multi':
        # Special case for multi-source streamer
        source_streamers = config.get('streamers', [])
        return streamer_class(source_streamers)
    else:
        return streamer_class(config)
