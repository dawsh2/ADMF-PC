"""
Async market data feed.

Async market data streaming and management.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json

from ..types import Bar
from ..async_protocols import MarketDataFeed

logger = logging.getLogger(__name__)


class AsyncMarketDataFeed:
    """
    Async market data feed implementation.
    
    Provides real-time market data streaming with
    subscription management and data callbacks.
    """
    
    def __init__(
        self,
        feed_name: str = "live_feed",
        websocket_url: str = "",
        api_key: str = "",
        data_callback: Optional[Callable] = None
    ):
        self.feed_name = feed_name
        self.websocket_url = websocket_url
        self.api_key = api_key
        self.data_callback = data_callback
        
        self.logger = logger.getChild(feed_name)
        
        # Connection state
        self._connected = False
        self._websocket = None
        
        # Subscriptions
        self._subscribed_symbols: set = set()
        self._subscription_requests: List[Dict[str, Any]] = []
        
        # Data cache
        self._latest_bars: Dict[str, Bar] = {}
        self._latest_quotes: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
    
    async def connect(self) -> None:
        """Connect to market data feed."""
        if self._connected:
            return
        
        self.logger.info(f"Connecting to market data feed: {self.feed_name}")
        
        try:
            await self._establish_websocket_connection()
            await self._authenticate()
            
            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_data())
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
            
            self._connected = True
            self.logger.info("Market data feed connected successfully")
            
            # Re-subscribe to symbols if any
            if self._subscribed_symbols:
                await self._resubscribe_all()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to market data feed: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from market data feed."""
        if not self._connected:
            return
        
        self.logger.info("Disconnecting from market data feed")
        
        # Cancel background tasks
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close websocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        self._connected = False
        self.logger.info("Market data feed disconnected")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to symbols."""
        if not symbols:
            return
        
        new_symbols = [s for s in symbols if s not in self._subscribed_symbols]
        if not new_symbols:
            return
        
        self.logger.info(f"Subscribing to symbols: {new_symbols}")
        
        # Add to subscription set
        self._subscribed_symbols.update(new_symbols)
        
        # Send subscription request if connected
        if self._connected:
            await self._send_subscription(new_symbols, action="subscribe")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        if not symbols:
            return
        
        symbols_to_remove = [s for s in symbols if s in self._subscribed_symbols]
        if not symbols_to_remove:
            return
        
        self.logger.info(f"Unsubscribing from symbols: {symbols_to_remove}")
        
        # Remove from subscription set
        self._subscribed_symbols.difference_update(symbols_to_remove)
        
        # Send unsubscription request if connected
        if self._connected:
            await self._send_subscription(symbols_to_remove, action="unsubscribe")
        
        # Clean up cached data
        for symbol in symbols_to_remove:
            self._latest_bars.pop(symbol, None)
            self._latest_quotes.pop(symbol, None)
    
    async def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest bar data for symbol."""
        bar = self._latest_bars.get(symbol)
        if bar:
            return bar.to_dict()
        return None
    
    async def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest quote data for symbol."""
        return self._latest_quotes.get(symbol)
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of subscribed symbols."""
        return list(self._subscribed_symbols)
    
    def is_connected(self) -> bool:
        """Check if connected to feed."""
        return self._connected
    
    async def _establish_websocket_connection(self) -> None:
        """Establish WebSocket connection."""
        # Placeholder - actual implementation would use aiohttp or websockets
        # This is a mock implementation
        self.logger.info("WebSocket connection established (mock)")
    
    async def _authenticate(self) -> None:
        """Authenticate with market data feed."""
        if not self.api_key:
            self.logger.warning("No API key provided for market data feed")
            return
        
        # Placeholder - send authentication message
        auth_message = {
            "action": "auth",
            "key": self.api_key
        }
        
        await self._send_message(auth_message)
        self.logger.info("Market data feed authentication sent")
    
    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send message to WebSocket."""
        if not self._websocket:
            self.logger.warning("WebSocket not connected, cannot send message")
            return
        
        try:
            # Placeholder - actual implementation would send via websocket
            message_str = json.dumps(message)
            self.logger.debug(f"Sending message: {message_str}")
            # await self._websocket.send(message_str)
            
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
    
    async def _send_subscription(self, symbols: List[str], action: str) -> None:
        """Send subscription/unsubscription request."""
        subscription_message = {
            "action": action,
            "bars": symbols,
            "quotes": symbols,
            "trades": symbols
        }
        
        await self._send_message(subscription_message)
        self.logger.debug(f"Sent {action} request for symbols: {symbols}")
    
    async def _resubscribe_all(self) -> None:
        """Re-subscribe to all symbols after reconnection."""
        if self._subscribed_symbols:
            await self._send_subscription(list(self._subscribed_symbols), "subscribe")
            self.logger.info(f"Re-subscribed to {len(self._subscribed_symbols)} symbols")
    
    async def _receive_data(self) -> None:
        """Continuously receive and process data from WebSocket."""
        self.logger.debug("Starting data receive task")
        
        while self._connected:
            try:
                # Placeholder - actual implementation would receive from websocket
                # message = await self._websocket.recv()
                # data = json.loads(message)
                
                # Mock data for demonstration
                await asyncio.sleep(1)  # Simulate data arrival
                
                # Process mock bar data
                for symbol in self._subscribed_symbols:
                    mock_bar = self._create_mock_bar(symbol)
                    await self._process_bar_data(mock_bar)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error receiving data: {e}")
                # Attempt reconnection
                await self._attempt_reconnection()
        
        self.logger.debug("Data receive task stopped")
    
    async def _process_bar_data(self, bar_data: Dict[str, Any]) -> None:
        """Process incoming bar data."""
        try:
            symbol = bar_data["symbol"]
            
            # Create Bar object
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.fromisoformat(bar_data["timestamp"]),
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=bar_data["volume"]
            )
            
            # Update cache
            self._latest_bars[symbol] = bar
            
            # Call data callback if provided
            if self.data_callback:
                await self.data_callback("bar", bar_data)
            
        except Exception as e:
            self.logger.error(f"Error processing bar data: {e}")
    
    async def _process_quote_data(self, quote_data: Dict[str, Any]) -> None:
        """Process incoming quote data."""
        try:
            symbol = quote_data["symbol"]
            
            # Update cache
            self._latest_quotes[symbol] = quote_data
            
            # Call data callback if provided
            if self.data_callback:
                await self.data_callback("quote", quote_data)
            
        except Exception as e:
            self.logger.error(f"Error processing quote data: {e}")
    
    async def _heartbeat(self) -> None:
        """Send periodic heartbeat messages."""
        self.logger.debug("Starting heartbeat task")
        
        while self._connected:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self._connected:
                    heartbeat_message = {"action": "heartbeat"}
                    await self._send_message(heartbeat_message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat: {e}")
        
        self.logger.debug("Heartbeat task stopped")
    
    async def _attempt_reconnection(self) -> None:
        """Attempt to reconnect to market data feed."""
        self.logger.warning("Attempting to reconnect to market data feed")
        
        for attempt in range(self.reconnect_attempts):
            try:
                await asyncio.sleep(self.reconnect_delay)
                
                # Close existing connection
                if self._websocket:
                    await self._websocket.close()
                
                # Re-establish connection
                await self._establish_websocket_connection()
                await self._authenticate()
                await self._resubscribe_all()
                
                self.logger.info("Successfully reconnected to market data feed")
                return
                
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        # If all attempts failed
        self.logger.error("All reconnection attempts failed")
        self._connected = False
    
    def _create_mock_bar(self, symbol: str) -> Dict[str, Any]:
        """Create mock bar data for demonstration."""
        import random
        
        base_price = 100.0  # Mock base price
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "open": base_price + random.uniform(-1, 1),
            "high": base_price + random.uniform(0, 2),
            "low": base_price + random.uniform(-2, 0),
            "close": base_price + random.uniform(-1, 1),
            "volume": random.randint(1000, 10000)
        }