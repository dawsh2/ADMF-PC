"""
Live Data Handler for real-time market data streaming.
Adapts the AlpacaWebSocketStreamer to work with ADMF-PC's data handler protocol.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime
import pandas as pd

from .models import Bar
from .streamers.alpaca_streamer import AlpacaWebSocketStreamer, AlpacaConfig

logger = logging.getLogger(__name__)


class LiveDataHandler:
    """
    Live data handler that adapts AlpacaWebSocketStreamer to ADMF-PC data handler protocol.
    This provides real-time market data for live trading.
    """
    
    def __init__(self, handler_id: str, symbols: List[str], live_config: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize live data handler.
        
        Args:
            handler_id: Unique identifier for this handler
            symbols: List of symbols to stream (e.g., ['SPY'])
            live_config: Live trading configuration with API credentials
            config: Full component configuration
        """
        import os
        
        self.handler_id = handler_id
        self.symbols = symbols
        self.config = config
        self.live_config = live_config
        
        # Get API credentials from live_config or environment variables
        api_key = live_config.get('api_key') or os.getenv('ALPACA_API_KEY')
        secret_key = live_config.get('secret_key') or os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')
        
        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API credentials are required for live trading. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables, "
                "or pass them in live_config."
            )
        
        # Initialize the Alpaca WebSocket streamer
        feed = live_config.get('feed', 'iex')
        paper_trading = live_config.get('paper_trading', True)
        
        alpaca_config = AlpacaConfig(
            api_key=api_key,
            secret_key=secret_key,
            symbols=symbols,
            paper_trading=paper_trading,
            feed=feed
        )
        self.streamer = AlpacaWebSocketStreamer(alpaca_config)
        
        # Track streaming state
        self._is_streaming = False
        self._bar_count = 0
        self._running = False
        
        # Container and event bus for event publication (set by container)
        self.container = None
        self.event_bus = None
        
        logger.info(f"LiveDataHandler initialized for symbols: {symbols}")
    
    def set_container(self, container) -> None:
        """Set the container reference for event publishing."""
        self.container = container
    
    def execute(self) -> None:
        """
        Execute live data streaming - called during container execution phase.
        
        Streams live market data through the event system, enabling event-driven
        execution where other components react naturally to BAR events.
        """
        logger.info(f"🔴 Live data handler execute() called for symbols: {self.symbols}")
        
        if not self._running:
            self.start_streaming()
        
        # Stream live bars using async streaming
        bars_streamed = 0
        max_bars = getattr(self, 'max_bars', float('inf'))
        
        # Handle None max_bars
        if max_bars is None:
            max_bars = float('inf')
        
        logger.info(f"🔴 Starting live bar streaming, max_bars: {max_bars}, symbols: {self.symbols}")
        
        try:
            # Run the async streaming in the current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def stream_bars():
                nonlocal bars_streamed
                
                # Connect to Alpaca WebSocket (paid account - should work)
                logger.info("🔴 Connecting to Alpaca WebSocket (paid account)...")
                connected = await self.streamer.start()
                
                if not connected:
                    logger.error("❌ Failed to connect to Alpaca WebSocket")
                    # For paid accounts, let's try one retry with longer delay
                    logger.info("⚠️ Retrying connection in 5 seconds...")
                    await asyncio.sleep(5)
                    connected = await self.streamer.start()
                    
                    if not connected:
                        logger.error("❌ Failed to connect after retry. Check Alpaca credentials and account status.")
                        logger.error("💡 Troubleshooting suggestions:")
                        logger.error("1. Close any other Alpaca WebSocket connections")
                        logger.error("2. Wait a few minutes and try again")
                        logger.error("3. Check if account has WebSocket permissions")
                        logger.error("4. Try during different market hours")
                        return
                
                logger.info("✅ Connected to Alpaca WebSocket, starting bar streaming...")
                
                # Stream real live data
                async for timestamp, bars_dict in self.streamer.stream_bars():
                    if bars_streamed >= max_bars:
                        break
                    
                    # Process each bar in the dict (usually just one)
                    for symbol, bar in bars_dict.items():
                        # Publish BAR event using the same pattern as SimpleHistoricalDataHandler
                        self._publish_bar_event(bar)
                        bars_streamed += 1
                    
                    # Progress indicator
                    if bars_streamed % 10 == 0:
                        logger.info(f"📊 Live bars streamed: {bars_streamed}")
                    
                    # For demo purposes, limit to prevent infinite streaming
                    if bars_streamed >= 100:  # Stream up to 100 bars for demo
                        logger.info("🔴 Demo limit reached, stopping live stream")
                        break
                
                # Disconnect when done
                await self.streamer.stop()
            
            # Run the streaming
            loop.run_until_complete(stream_bars())
            
        except Exception as e:
            logger.error(f"Error in live data streaming: {e}")
            logger.error("This may be due to:")
            logger.error("1. Network connectivity issues")
            logger.error("2. Invalid Alpaca credentials")
            logger.error("3. WebSocket connection issues during market hours")
        finally:
            if 'loop' in locals():
                loop.close()
            
        logger.info(f"✅ Live data handler completed: streamed {bars_streamed:,} live bars")
    
    def _publish_bar_event(self, alpaca_bar) -> None:
        """
        Publish BAR event to the event bus following ADMF-PC patterns.
        
        Args:
            alpaca_bar: Bar from Alpaca WebSocket stream
        """
        # Convert Alpaca bar to ADMF-PC Bar format
        bar = Bar(
            symbol=alpaca_bar.symbol,
            timestamp=alpaca_bar.timestamp,
            open=alpaca_bar.open,
            high=alpaca_bar.high,
            low=alpaca_bar.low,
            close=alpaca_bar.close,
            volume=alpaca_bar.volume,
            metadata={
                'vwap': getattr(alpaca_bar, 'vwap', None),
                'trade_count': getattr(alpaca_bar, 'trade_count', None)
            }
        )
        
        # Publish BAR event to event bus (same pattern as SimpleHistoricalDataHandler)
        if self.container:
            from ..core.events.types import Event, EventType
            event = Event(
                event_type=EventType.BAR.value,
                payload={
                    'symbol': bar.symbol,
                    'timestamp': bar.timestamp,
                    'bar': bar,
                    'original_bar_index': self._bar_count,  # Use bar count as index for live data
                    'split_bar_index': self._bar_count      # Same for live data
                },
                source_id=f"live_data_{bar.symbol}",
                container_id=self.container.container_id if self.container else None
            )
            logger.debug(f"🔴 Publishing live BAR event #{self._bar_count} for {bar.symbol} at {bar.timestamp}")
            self.container.event_bus.publish(event)
            self._bar_count += 1
        else:
            logger.warning("No container set - cannot publish BAR events")
    
    async def _simulate_live_data(self) -> None:
        """
        Simulate live data by generating realistic bar data for demonstration.
        This is used when Alpaca WebSocket connection fails due to API limits.
        """
        import random
        from datetime import datetime, timezone
        
        logger.info("🎭 Starting simulated live data stream...")
        
        # Starting price around SPY's typical range
        base_price = 575.0
        current_price = base_price
        
        # Generate 20 simulated bars to demonstrate the system
        for i in range(20):
            # Simulate realistic price movement
            price_change = random.uniform(-0.5, 0.5)  # ±$0.50 movement
            current_price += price_change
            
            # Create realistic OHLC data with proper validation
            open_price = current_price - random.uniform(-0.2, 0.2)
            close_price = current_price
            high = max(open_price, close_price) + random.uniform(0, 0.3)
            low = min(open_price, close_price) - random.uniform(0, 0.3)
            volume = random.randint(1000000, 5000000)
            
            # Create simulated bar
            timestamp = datetime.now(timezone.utc)
            
            # Create a mock Alpaca bar object
            class MockAlpacaBar:
                def __init__(self, symbol, timestamp, open, high, low, close, volume):
                    self.symbol = symbol
                    self.timestamp = timestamp
                    self.open = open
                    self.high = high
                    self.low = low
                    self.close = close
                    self.volume = volume
                    self.vwap = (high + low + close) / 3
                    self.trade_count = random.randint(500, 1500)
            
            mock_bar = MockAlpacaBar(
                symbol=self.symbols[0],
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume
            )
            
            # Publish the simulated bar
            self._publish_bar_event(mock_bar)
            
            logger.info(f"🎭 Simulated bar {i+1}/20: {self.symbols[0]} @ ${close_price:.2f} (Vol: {volume:,})")
            
            # Wait between bars to simulate real-time streaming
            await asyncio.sleep(1)  # 1 second between bars for demo
        
        logger.info("🎭 Simulated live data stream completed")
    
    def get_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get streaming data. For live data, this starts streaming and returns bars as they arrive.
        This method is called by ADMF-PC's data loading framework.
        
        Args:
            symbols: Optional symbols filter (uses self.symbols if None)
            
        Returns:
            DataFrame with OHLCV bar data
        """
        if symbols is None:
            symbols = self.symbols
            
        logger.info(f"🔴 Starting live data stream for {symbols}")
        
        # Start the streaming process
        try:
            # Run the async streaming in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start streaming and collect bars
            bars_data = []
            
            async def collect_bars():
                async for bar in self.streamer.stream_bars():
                    self._bar_count += 1
                    
                    # Convert Alpaca bar to ADMF-PC Bar format
                    admf_bar = Bar(
                        timestamp=bar.timestamp,
                        symbol=bar.symbol,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        vwap=bar.vwap,
                        trade_count=bar.trade_count
                    )
                    
                    bars_data.append({
                        'timestamp': admf_bar.timestamp,
                        'symbol': admf_bar.symbol,
                        'open': admf_bar.open,
                        'high': admf_bar.high,
                        'low': admf_bar.low,
                        'close': admf_bar.close,
                        'volume': admf_bar.volume,
                        'vwap': admf_bar.vwap,
                        'trade_count': admf_bar.trade_count
                    })
                    
                    logger.info(f"📊 Live bar received: {admf_bar.symbol} @ {admf_bar.timestamp} - Close: ${admf_bar.close:.2f}")
                    
                    # For demo purposes, yield bars as they arrive
                    # In a real implementation, you might want to batch them or use a different approach
                    if len(bars_data) >= 100:  # Return data in batches
                        df = pd.DataFrame(bars_data)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        return df
            
            # Run the streaming collection
            result = loop.run_until_complete(collect_bars())
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in live data streaming: {e}")
            return pd.DataFrame()
        finally:
            loop.close()
    
    async def stream_live_data(self) -> AsyncIterator[Bar]:
        """
        Async generator for streaming live data.
        This is a more advanced interface for real-time applications.
        """
        async for bar in self.streamer.stream_bars():
            yield Bar(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap,
                trade_count=bar.trade_count
            )
    
    def start_streaming(self) -> None:
        """Start the live data stream."""
        self._is_streaming = True
        self._running = True
        logger.info(f"🔴 Live streaming started for {self.symbols}")
    
    def stop_streaming(self) -> None:
        """Stop the live data stream."""
        self._is_streaming = False
        self._running = False
        if hasattr(self.streamer, 'disconnect'):
            asyncio.run(self.streamer.disconnect())
        logger.info("🔴 Live streaming stopped")
    
    def has_more_data(self) -> bool:
        """Check if there's more data available (always True for live streaming)."""
        return self._running and self._is_streaming
    
    def start(self) -> None:
        """Start data streaming (alias for start_streaming)."""
        self.start_streaming()
    
    def stop(self) -> None:
        """Stop data streaming (alias for stop_streaming).""" 
        self.stop_streaming()
    
    @property
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._is_streaming
    
    @property
    def bar_count(self) -> int:
        """Get number of bars received."""
        return self._bar_count