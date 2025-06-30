"""
Alpaca trade updates WebSocket stream.

Real-time order and fill notifications via WebSocket.
Clean async patterns, no complex abstractions.
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Any, Optional, AsyncIterator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TradeUpdateType(Enum):
    """Types of trade updates from Alpaca."""
    NEW = "new"
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    CANCELED = "canceled"
    EXPIRED = "expired"
    DONE_FOR_DAY = "done_for_day"
    REPLACED = "replaced"
    REJECTED = "rejected"
    PENDING_NEW = "pending_new"
    STOPPED = "stopped"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    CALCULATED = "calculated"
    SUSPENDED = "suspended"
    ORDER_REPLACE_REJECTED = "order_replace_rejected"
    ORDER_CANCEL_REJECTED = "order_cancel_rejected"


@dataclass
class TradeUpdate:
    """Parsed trade update from Alpaca WebSocket."""
    event: TradeUpdateType
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    filled_quantity: float
    price: Optional[float]
    filled_avg_price: Optional[float]
    status: str
    timestamp: datetime
    raw_data: Dict[str, Any]


class AlpacaTradeStream:
    """
    Alpaca trade updates WebSocket stream.
    
    Provides real-time order and fill updates.
    Separate from market data stream.
    """
    
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_trading = paper_trading
        
        # WebSocket URL for trade updates
        if paper_trading:
            self.ws_url = "wss://paper-api.alpaca.markets/stream"
        else:
            self.ws_url = "wss://api.alpaca.markets/stream"
        
        self._websocket = None
        self._connected = False
        self._authenticated = False
        self.logger = logger.getChild("trade_stream")
        
    async def connect(self) -> bool:
        """Connect and authenticate to trade updates stream."""
        try:
            self.logger.info(f"Connecting to Alpaca trade stream: {self.ws_url}")
            
            # Connect to WebSocket
            self._websocket = await websockets.connect(self.ws_url)
            self._connected = True
            
            # Wait for welcome message
            welcome = await self._websocket.recv()
            welcome_data = json.loads(welcome)
            self.logger.debug(f"Welcome message: {welcome_data}")
            
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            
            await self._websocket.send(json.dumps(auth_msg))
            
            # Wait for auth response
            auth_response = await self._websocket.recv()
            auth_data = json.loads(auth_response)
            
            if self._is_authenticated(auth_data):
                self._authenticated = True
                self.logger.info("Successfully authenticated to trade stream")
                
                # Subscribe to trade updates
                await self._subscribe()
                return True
            else:
                self.logger.error(f"Authentication failed: {auth_data}")
                await self.disconnect()
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to trade stream: {e}")
            await self.disconnect()
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from trade stream."""
        self._authenticated = False
        self._connected = False
        
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None
        
        self.logger.info("Disconnected from trade stream")
    
    async def _subscribe(self) -> None:
        """Subscribe to trade updates."""
        sub_msg = {
            "action": "listen",
            "data": {
                "streams": ["trade_updates"]
            }
        }
        
        await self._websocket.send(json.dumps(sub_msg))
        self.logger.info("Subscribed to trade updates")
    
    def _is_authenticated(self, data: Any) -> bool:
        """Check if authentication was successful."""
        if isinstance(data, dict):
            # Check for direct auth success
            if data.get("status") == "authorized":
                return True
            # Check for message in data
            if data.get("data", {}).get("status") == "authorized":
                return True
        return False
    
    async def stream_updates(self) -> AsyncIterator[TradeUpdate]:
        """
        Stream trade updates as they arrive.
        
        Yields TradeUpdate objects for each order event.
        """
        if not self._authenticated:
            raise Exception("Not authenticated to trade stream")
        
        self.logger.info("Starting to stream trade updates...")
        
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    
                    # Skip non-trade update messages
                    if not isinstance(data, dict) or data.get("stream") != "trade_updates":
                        continue
                    
                    # Parse trade update
                    update = self._parse_trade_update(data.get("data", {}))
                    if update:
                        yield update
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to decode message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing trade update: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("Trade stream connection closed")
            self._connected = False
            self._authenticated = False
        except Exception as e:
            self.logger.error(f"Trade stream error: {e}")
            raise
    
    def _parse_trade_update(self, data: Dict[str, Any]) -> Optional[TradeUpdate]:
        """Parse raw trade update into TradeUpdate object."""
        try:
            event = data.get("event", "").lower()
            if not event:
                return None
            
            # Parse timestamp
            timestamp_str = data.get("timestamp", "")
            timestamp = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00")
            ) if timestamp_str else datetime.now()
            
            # Create update object
            order_data = data.get("order", {})
            
            return TradeUpdate(
                event=TradeUpdateType(event),
                order_id=order_data.get("id", ""),
                client_order_id=order_data.get("client_order_id", ""),
                symbol=order_data.get("symbol", ""),
                side=order_data.get("side", ""),
                order_type=order_data.get("order_type", ""),
                quantity=float(order_data.get("qty", 0)),
                filled_quantity=float(order_data.get("filled_qty", 0)),
                price=float(order_data.get("limit_price")) if order_data.get("limit_price") else None,
                filled_avg_price=float(order_data.get("filled_avg_price")) if order_data.get("filled_avg_price") else None,
                status=order_data.get("status", ""),
                timestamp=timestamp,
                raw_data=data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse trade update: {e}")
            return None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        return self._connected and self._authenticated


# Example usage demonstrating clean async patterns
async def demo_trade_stream(api_key: str, secret_key: str):
    """
    Demo function showing how to use trade stream.
    
    This demonstrates the clean async pattern - no callbacks,
    just async iteration over updates as they arrive.
    """
    stream = AlpacaTradeStream(api_key, secret_key, paper_trading=True)
    
    try:
        # Connect to stream
        if not await stream.connect():
            print("Failed to connect to trade stream")
            return
        
        print("Connected! Listening for trade updates...")
        print("-" * 60)
        
        # Stream updates
        async for update in stream.stream_updates():
            print(f"[{update.timestamp.strftime('%H:%M:%S')}] "
                  f"{update.event.value.upper()}: "
                  f"{update.symbol} {update.side} "
                  f"{update.filled_quantity}/{update.quantity} "
                  f"@ ${update.filled_avg_price or 'pending'}")
            
            # React to specific events
            if update.event == TradeUpdateType.FILL:
                print(f"  🎯 Order filled! {update.symbol} "
                      f"{update.filled_quantity} @ ${update.filled_avg_price}")
            elif update.event == TradeUpdateType.REJECTED:
                print(f"  ❌ Order rejected! Check raw data: {update.raw_data}")
            
    except KeyboardInterrupt:
        print("\nStopping trade stream...")
    finally:
        await stream.disconnect()


if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    
    if api_key and secret_key:
        asyncio.run(demo_trade_stream(api_key, secret_key))
    else:
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")