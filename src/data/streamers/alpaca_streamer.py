"""
Alpaca WebSocket streaming implementation following ADMF-PC patterns.

Live data streaming using Protocol+Composition - NO INHERITANCE!
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Any, Optional, AsyncIterator, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from decimal import Decimal

from ..models import Bar
from .bar_streamer import StreamedBar

logger = logging.getLogger(__name__)


@dataclass
class AlpacaConfig:
    """Configuration for Alpaca WebSocket streaming."""
    api_key: str
    secret_key: str
    symbols: List[str]
    paper_trading: bool = True
    timeframe: str = "1Min"  # 1Min, 5Min, 15Min, 1Hour, 1Day, or "tick"
    feed: str = "iex"  # iex or sip


class AlpacaWebSocketStreamer:
    """
    Alpaca WebSocket streamer - NO INHERITANCE!
    
    Implements streaming protocols through duck typing.
    Follows ADMF-PC composition patterns.
    """
    
    def __init__(self, config: AlpacaConfig):
        self.config = config
        self._websocket = None
        self._connected = False
        self._subscribed = False
        
        # Latest bars cache
        self.latest_bars: Dict[str, StreamedBar] = {}
        
        # WebSocket URLs - use v1beta3 for crypto, v2 for stocks
        is_crypto = any('/' in symbol or symbol in ['BTCUSD', 'ETHUSD'] for symbol in config.symbols)
        
        if is_crypto:
            # Use v1beta3 endpoint for crypto
            self.ws_url = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
            logger.info(f"Using v1beta3 crypto endpoint for {config.symbols}")
        else:
            # Use v2 endpoint for stocks
            self.ws_url = "wss://stream.data.alpaca.markets/v2/iex" if config.feed == "iex" else "wss://stream.data.alpaca.markets/v2/sip"
            logger.info(f"Using v2 {config.feed} endpoint for {config.symbols}")
        
        # Authentication payload
        self.auth_payload = {
            "action": "auth",
            "key": config.api_key,
            "secret": config.secret_key
        }
        
        # Subscription payload - subscribe to trades for tick data, bars otherwise
        logger.info(f"AlpacaWebSocketStreamer timeframe: {config.timeframe}")
        if config.timeframe.lower() == "tick":
            self.subscribe_payload = {
                "action": "subscribe",
                "trades": config.symbols
            }
            self.stream_type = "trades"
            logger.info(f"Subscribing to TRADES for tick data")
        else:
            self.subscribe_payload = {
                "action": "subscribe",
                "bars": config.symbols
            }
            self.stream_type = "bars"
            logger.info(f"Subscribing to BARS for {config.timeframe} data")
        
        logger.info(f"AlpacaWebSocketStreamer initialized for symbols: {config.symbols}")
        logger.info(f"WebSocket URL: {self.ws_url}")
        logger.info(f"Feed: {config.feed}, Paper Trading: {config.paper_trading}")
    
    @property
    def name(self) -> str:
        return "alpaca_websocket_streamer"
    
    @property
    def connected(self) -> bool:
        return self._connected and self._subscribed
    
    # Implements HasLifecycle protocol
    async def start(self) -> bool:
        """Start WebSocket connection and subscribe to symbols."""
        try:
            logger.info(f"Connecting to Alpaca WebSocket: {self.ws_url}")
            
            # Connect to WebSocket
            self._websocket = await websockets.connect(self.ws_url)
            self._connected = True
            
            # Authenticate
            await self._authenticate()
            
            # Subscribe to symbols
            await self._subscribe()
            
            logger.info(f"Successfully connected and subscribed to: {self.config.symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Alpaca WebSocket: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> None:
        """Stop WebSocket connection."""
        self._connected = False
        self._subscribed = False
        
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None
        
        logger.info("Alpaca WebSocket connection stopped")
    
    async def _authenticate(self) -> None:
        """Authenticate with Alpaca WebSocket."""
        logger.debug(f"Sending auth payload: {self.auth_payload}")
        await self._websocket.send(json.dumps(self.auth_payload))
        
        # Wait for auth response
        response = await self._websocket.recv()
        logger.info(f"Received auth response: {response[:200]}...")  # Log first 200 chars
        auth_data = json.loads(response)
        logger.info(f"Parsed auth data: {auth_data}, type: {type(auth_data)}")
        
        # Handle response as list or single message
        if isinstance(auth_data, list):
            auth_data = auth_data[0] if auth_data else {}
        
        if auth_data.get("T") == "success" and auth_data.get("msg") == "authenticated":
            logger.info("Alpaca WebSocket authentication successful")
        elif auth_data.get("T") == "success" and auth_data.get("msg") == "connected":
            # Initial connection message - need to wait for actual auth response
            logger.info("Connected to Alpaca WebSocket, waiting for authentication...")
            auth_response = await self._websocket.recv()
            auth_data = json.loads(auth_response)
            if isinstance(auth_data, list):
                auth_data = auth_data[0] if auth_data else {}
            
            if auth_data.get("T") == "success" and auth_data.get("msg") == "authenticated":
                logger.info("Alpaca WebSocket authentication successful")
            else:
                raise Exception(f"Authentication failed: {auth_data}")
        else:
            raise Exception(f"Authentication failed: {auth_data}")
    
    async def _subscribe(self) -> None:
        """Subscribe to bar data for configured symbols."""
        logger.info(f"Sending subscribe payload: {self.subscribe_payload}")
        await self._websocket.send(json.dumps(self.subscribe_payload))
        
        # Wait for subscription response
        response = await self._websocket.recv()
        logger.info(f"Received subscribe response: {response[:200]}...")
        sub_data = json.loads(response)
        logger.debug(f"Parsed subscribe data: {sub_data}, type: {type(sub_data)}")
        
        # Handle response as list or single message
        if isinstance(sub_data, list):
            sub_data = sub_data[0] if sub_data else {}
        
        if sub_data.get("T") == "subscription":
            self._subscribed = True
            if self.stream_type == "trades":
                logger.info(f"Successfully subscribed to trades: {sub_data.get('trades', [])}")
            else:
                logger.info(f"Successfully subscribed to bars: {sub_data.get('bars', [])}")
        else:
            raise Exception(f"Subscription failed: {sub_data}")
    
    async def stream_bars(self) -> AsyncIterator[Tuple[datetime, Dict[str, StreamedBar]]]:
        """
        Stream live bars from Alpaca WebSocket.
        
        Yields:
            Tuple of (timestamp, bars) where bars is a dict of symbol -> StreamedBar
        """
        if not self.connected:
            raise Exception("Not connected to Alpaca WebSocket")
        
        logger.info("Starting to stream live bars from Alpaca...")
        
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle different message types
                    if isinstance(data, list):
                        # Multiple messages in array
                        for msg in data:
                            bar = self._process_message(msg)
                            if bar:
                                # Yield immediately when we get a bar
                                yield bar.timestamp, {bar.symbol: bar}
                    else:
                        # Single message
                        bar = self._process_message(data)
                        if bar:
                            yield bar.timestamp, {bar.symbol: bar}
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Alpaca WebSocket connection closed")
            self._connected = False
            self._subscribed = False
        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}")
            raise
    
    def _process_message(self, msg: Dict[str, Any]) -> Optional[StreamedBar]:
        """Process a single WebSocket message and return StreamedBar if it's bar or trade data."""
        try:
            msg_type = msg.get("T")
            
            # Check if this is a bar message
            if msg_type == "b":  # Minute bar data
                return self._process_bar_message(msg)
            elif msg_type == "d":  # Daily bar data (arrives every minute)
                logger.info(f"Processing daily bar as minute bar (Alpaca sends these when no trades)")
                return self._process_bar_message(msg)  # Process daily bars same as minute bars
            elif msg_type == "t":  # Trade/tick data
                return self._process_trade_message(msg)
            else:
                return None
            
        except Exception as e:
            logger.warning(f"Error processing message: {e}")
            return None
    
    def _process_bar_message(self, msg: Dict[str, Any]) -> Optional[StreamedBar]:
        """Process a bar message and return StreamedBar."""
        try:
            # Extract bar data
            symbol = msg.get("S")
            if not symbol:
                return None
            
            # Parse timestamp - Alpaca sends in RFC3339 format
            timestamp_str = msg.get("t")
            if not timestamp_str:
                return None
            
            # Convert to datetime
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            
            # Create StreamedBar
            bar = StreamedBar(
                timestamp=timestamp,
                symbol=symbol,
                open=float(msg.get("o", 0)),
                high=float(msg.get("h", 0)),
                low=float(msg.get("l", 0)),
                close=float(msg.get("c", 0)),
                volume=float(msg.get("v", 0))
            )
            
            # Update cache
            self.latest_bars[symbol] = bar
            
            # Log bar - always log even with 0 volume
            logger.info(f"📊 Received bar: {symbol} {timestamp.strftime('%H:%M:%S')} O:${bar.open} H:${bar.high} L:${bar.low} C:${bar.close} V:{bar.volume} (Alpaca exchange only)")
            
            return bar
            
        except Exception as e:
            logger.warning(f"Error processing bar message: {e}")
            return None
    
    def _process_trade_message(self, msg: Dict[str, Any]) -> Optional[StreamedBar]:
        """Process a trade/tick message and convert to StreamedBar format."""
        try:
            # Extract trade data
            symbol = msg.get("S")
            if not symbol:
                return None
            
            # Parse timestamp
            timestamp_str = msg.get("t")
            if not timestamp_str:
                return None
            
            # Convert to datetime
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            
            # For tick data, create a "bar" with all prices set to trade price
            price = float(msg.get("p", 0))
            size = float(msg.get("s", 0))
            
            # Create StreamedBar from tick
            bar = StreamedBar(
                timestamp=timestamp,
                symbol=symbol,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=size  # Trade size as volume
            )
            
            # Update cache
            self.latest_bars[symbol] = bar
            
            # Log tick
            logger.info(f"📊 Received tick: {symbol} @ ${price} Size:{size} Time:{timestamp.strftime('%H:%M:%S.%f')[:-3]}")
            
            return bar
            
        except Exception as e:
            logger.warning(f"Error processing trade message: {e}")
            return None
    
    # Implements DataProvider-like functionality
    def get_latest_bar(self, symbol: str) -> Optional[StreamedBar]:
        """Get latest bar for symbol."""
        return self.latest_bars.get(symbol)
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get currently subscribed symbols."""
        return self.config.symbols.copy()
    
    # Implements streaming capabilities
    def has_data(self, symbol: str) -> bool:
        """Check if we have data for symbol."""
        return symbol in self.latest_bars
    
    async def wait_for_bar(self, symbol: str, timeout: float = 60.0) -> Optional[StreamedBar]:
        """Wait for next bar for specific symbol."""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if symbol in self.latest_bars:
                return self.latest_bars[symbol]
            await asyncio.sleep(0.1)
        
        return None


class AlpacaBarPrinter:
    """
    Simple console printer for Alpaca bars.
    
    Demonstrates how to consume the streamer.
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.bar_count = 0
        self.start_time = datetime.now()
    
    def print_bar(self, bar: StreamedBar) -> None:
        """Print bar to console with nice formatting."""
        self.bar_count += 1
        
        # Calculate time since start
        elapsed = datetime.now() - self.start_time
        
        print(f"[{elapsed}] #{self.bar_count:03d} | "
              f"{bar.symbol:4s} | "
              f"{bar.timestamp.strftime('%H:%M:%S')} | "
              f"O:{bar.open:7.2f} H:{bar.high:7.2f} L:{bar.low:7.2f} C:{bar.close:7.2f} | "
              f"V:{bar.volume:8,.0f}")
    
    async def run_with_streamer(self, streamer: AlpacaWebSocketStreamer) -> None:
        """Run the printer with a streamer."""
        print(f"\n{'='*80}")
        print(f"🔴 LIVE ALPACA BARS - {', '.join(self.symbols)}")
        print(f"{'='*80}")
        print(f"{'Time':>12} | {'#':>3} | {'Sym':>4} | {'Timestamp':>8} | {'OHLC':>35} | {'Volume':>12}")
        print(f"{'-'*80}")
        
        try:
            async for timestamp, bars in streamer.stream_bars():
                for symbol, bar in bars.items():
                    if symbol in self.symbols:
                        self.print_bar(bar)
                        
        except KeyboardInterrupt:
            print(f"\n\n📊 Session Summary:")
            print(f"  Total bars received: {self.bar_count}")
            print(f"  Runtime: {datetime.now() - self.start_time}")
            print(f"  Symbols: {', '.join(self.symbols)}")
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")


# Factory function following ADMF-PC patterns
def create_alpaca_streamer(
    api_key: str,
    secret_key: str,
    symbols: List[str],
    paper_trading: bool = True,
    timeframe: str = "1Min",
    feed: str = "iex"
) -> AlpacaWebSocketStreamer:
    """
    Factory function to create Alpaca WebSocket streamer.
    
    Args:
        api_key: Alpaca API key
        secret_key: Alpaca secret key
        symbols: List of symbols to stream
        paper_trading: Use paper trading endpoints
        timeframe: Bar timeframe (1Min, 5Min, etc.)
        feed: Data feed (iex or sip)
    
    Returns:
        Configured AlpacaWebSocketStreamer
    """
    config = AlpacaConfig(
        api_key=api_key,
        secret_key=secret_key,
        symbols=symbols,
        paper_trading=paper_trading,
        timeframe=timeframe,
        feed=feed
    )
    
    return AlpacaWebSocketStreamer(config)


async def demo_alpaca_streaming(api_key: str, secret_key: str, symbols: List[str]) -> None:
    """
    Demo function to show live Alpaca streaming.
    
    Usage:
        import asyncio
        from src.data.streamers.alpaca_streamer import demo_alpaca_streaming
        
        asyncio.run(demo_alpaca_streaming(
            api_key="your_api_key",
            secret_key="your_secret_key", 
            symbols=["SPY", "QQQ"]
        ))
    """
    # Create streamer
    streamer = create_alpaca_streamer(
        api_key=api_key,
        secret_key=secret_key,
        symbols=symbols,
        paper_trading=True  # Use paper trading
    )
    
    # Create printer
    printer = AlpacaBarPrinter(symbols)
    
    try:
        # Start streaming
        if await streamer.start():
            await printer.run_with_streamer(streamer)
        else:
            print("❌ Failed to start Alpaca streaming")
            
    finally:
        await streamer.stop()


if __name__ == "__main__":
    # Example usage - replace with your actual credentials
    import os
    
    api_key = os.getenv("ALPACA_API_KEY", "your_api_key_here")
    secret_key = os.getenv("ALPACA_API_SECRET", "your_secret_key_here")
    
    if api_key == "your_api_key_here":
        print("❌ Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        print("   export ALPACA_API_KEY=your_key")
        print("   export ALPACA_API_SECRET=your_secret")
    else:
        # Demo with SPY
        asyncio.run(demo_alpaca_streaming(
            api_key=api_key,
            secret_key=secret_key,
            symbols=["SPY"]
        ))