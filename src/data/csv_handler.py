"""
Simple CSV Data Handler for Symbol-Timeframe containers.

This is a minimal implementation focused on streaming CSV data
for the EVENT_FLOW_ARCHITECTURE.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Callable, AsyncGenerator
import logging
import asyncio
import pytz

from .models import Bar

logger = logging.getLogger(__name__)

# Pre-create timezone object for performance
ET_TZ = pytz.timezone('US/Eastern')


class CSVDataHandler:
    """
    Simple CSV data handler that streams bars to an event handler.
    
    Designed to work with SymbolTimeframeContainer in the new architecture.
    """
    
    def __init__(self, 
                 file_path: str,
                 symbol: str,
                 timeframe: str,
                 event_handler: Optional[Callable] = None,
                 max_bars: Optional[int] = None):
        """
        Initialize CSV data handler.
        
        Args:
            file_path: Path to CSV file
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1d')
            event_handler: Callback function for each bar
            max_bars: Optional limit on number of bars to process
        """
        self.file_path = Path(file_path)
        self.symbol = symbol
        self.timeframe = timeframe
        self.event_handler = event_handler
        self.max_bars = max_bars
        
        self.data = None
        self.current_index = 0
        self._streaming = False
        
    async def start_streaming(self):
        """Start streaming data."""
        # Load data if not already loaded
        if self.data is None:
            self._load_data()
            
        self._streaming = True
        logger.info(f"Started streaming {self.symbol} data from {self.file_path}")
        
        # Process data asynchronously with small delays
        # This allows other containers to initialize and subscribe
        total_bars = len(self.data)
        bars_to_process = self.max_bars if self.max_bars else total_bars
        bars_to_process = min(bars_to_process, total_bars)  # Don't exceed available data
        
        if self.max_bars:
            logger.info(f"Processing limited to {bars_to_process} bars (--bars {self.max_bars})")
        
        # Give other containers time to initialize
        await asyncio.sleep(0.1)
        
        while self._streaming and self.current_index < bars_to_process:
            bar = self._get_next_bar()
            if bar and self.event_handler:
                # Log data load if debug enabled
                logger.debug(
                    f"Data load: {self.symbol} bar {self.current_index}, price={bar.close}"
                )
                self.event_handler(bar)
            self.current_index += 1
            
            # Small delay every 10 bars to allow event processing
            if self.current_index % 10 == 0:
                await asyncio.sleep(0.001)
            
        logger.info(f"Finished streaming {self.symbol} data")
        
    async def stop_streaming(self):
        """Stop streaming data."""
        self._streaming = False
        logger.info(f"Stopped streaming {self.symbol} data")
        
    def _load_data(self):
        """Load CSV data into memory."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
            
        logger.info(f"Loading data from {self.file_path}")
        
        # Load CSV with basic columns
        self.data = pd.read_csv(
            self.file_path,
            index_col=0,
            parse_dates=True
        )
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            # Try uppercase columns
            self.data.columns = self.data.columns.str.lower()
            missing = [col for col in required_columns if col not in self.data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
        
        # Pre-calculate time-based features for performance
        self._calculate_time_features()
        
        logger.info(f"Loaded {len(self.data)} bars for {self.symbol}")
    
    def _calculate_time_features(self):
        """Pre-calculate time-based features to avoid repeated timezone conversions."""
        # Convert index to ET timezone
        if self.data.index.tz is None:
            # Assume UTC if no timezone
            self.data.index = self.data.index.tz_localize('UTC')
        
        # Convert to ET
        et_timestamps = self.data.index.tz_convert(ET_TZ)
        
        # Calculate time features
        self.data['et_hour'] = et_timestamps.hour
        self.data['et_minute'] = et_timestamps.minute
        self.data['et_time'] = self.data['et_hour'] * 100 + self.data['et_minute']
        
        # Intraday flag - market hours 9:30 AM - 3:55 PM ET
        self.data['intraday'] = (self.data['et_time'] >= 930) & (self.data['et_time'] < 1555)
        
        # Calculate bar_of_day for backward compatibility
        # Minutes since 9:30 AM divided by timeframe
        minutes_since_930 = ((self.data['et_hour'] - 9) * 60 + self.data['et_minute'] - 30)
        
        # Extract timeframe in minutes
        timeframe_minutes = 5  # Default for 5m
        if self.timeframe == '1m':
            timeframe_minutes = 1
        elif self.timeframe == '15m':
            timeframe_minutes = 15
        elif self.timeframe == '30m':
            timeframe_minutes = 30
        elif self.timeframe == '1h':
            timeframe_minutes = 60
        
        self.data['bar_of_day'] = minutes_since_930 // timeframe_minutes
        # Set pre-market bars to -1
        self.data.loc[self.data['et_time'] < 930, 'bar_of_day'] = -1
        
        logger.debug(f"Pre-calculated time features for {len(self.data)} bars")
        
    def _get_next_bar(self) -> Optional[Bar]:
        """Get the next bar from the data."""
        if self.current_index >= len(self.data):
            return None
            
        row = self.data.iloc[self.current_index]
        
        return Bar(
            symbol=self.symbol,
            timestamp=self.data.index[self.current_index],
            open=row['open'],
            high=row['high'], 
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )