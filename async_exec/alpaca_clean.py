"""
Clean Alpaca broker implementation following the async execution architecture.

Async at the boundaries (I/O operations), sync at the core.
No complex bridges - just clean async patterns.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Set, AsyncIterator
from datetime import datetime, timezone
from decimal import Decimal
from dataclasses import dataclass

from ...types import Order, Fill, Position, OrderSide, OrderType, FillStatus
from ...async_protocols import AsyncBroker
from .base import BrokerConfig, RateLimiter, CacheManager, ConnectionManager, OrderValidator
from .alpaca_trade_stream import AlpacaTradeStream, TradeUpdate, TradeUpdateType

logger = logging.getLogger(__name__)


@dataclass
class AlpacaConfig(BrokerConfig):
    """Alpaca-specific broker configuration."""
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    feed: str = "sip"  # sip or iex
    

class CleanAlpacaBroker:
    """
    Clean async Alpaca broker implementation.
    
    Follows the architecture principles:
    - Async for I/O operations (API calls, WebSocket)
    - Sync for data structures and calculations
    - No complex bridges or thread coordination
    - Clean boundaries between async and sync
    """
    
    def __init__(self, config: AlpacaConfig):
        self.config = config
        self.logger = logger.getChild("alpaca")
        
        # Composition: Use building blocks from base
        self.rate_limiter = RateLimiter(
            max_concurrent=config.rate_limit,
            min_interval=config.min_request_interval
        )
        self.cache = CacheManager(ttl=config.cache_ttl)
        self.connection = ConnectionManager("alpaca")
        self.validator = OrderValidator(
            supported_types=['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'],
            min_size=1.0
        )
        
        # HTTP session (created on connect)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket trade stream
        self._trade_stream: Optional[AlpacaTradeStream] = None
        
        # Order tracking
        self._pending_orders: Set[str] = set()
        self._order_map: Dict[str, str] = {}  # internal_id -> alpaca_id
        self._alpaca_to_internal: Dict[str, str] = {}  # alpaca_id -> internal_id
        
        # API headers
        self._headers = {
            "APCA-API-KEY-ID": config.api_key,
            "APCA-API-SECRET-KEY": config.secret_key,
            "Content-Type": "application/json"
        }
    
    # Implements AsyncBroker protocol through duck typing
    async def connect(self) -> None:
        """Connect to Alpaca API and trade stream."""
        # Connect HTTP session
        await self.connection.connect(
            self._create_session,
            self._authenticate
        )
        
        # Connect trade updates WebSocket
        try:
            self._trade_stream = AlpacaTradeStream(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper_trading=self.config.paper_trading
            )
            
            if await self._trade_stream.connect():
                self.logger.info("Connected to Alpaca trade updates WebSocket")
            else:
                self.logger.warning("Failed to connect trade stream, will use polling")
                self._trade_stream = None
                
        except Exception as e:
            self.logger.warning(f"Trade stream connection failed: {e}")
            self._trade_stream = None
    
    async def disconnect(self) -> None:
        """Disconnect from Alpaca API and trade stream."""
        # Disconnect trade stream
        if self._trade_stream:
            try:
                await self._trade_stream.disconnect()
            except Exception as e:
                self.logger.warning(f"Error disconnecting trade stream: {e}")
            self._trade_stream = None
        
        # Disconnect HTTP session
        await self.connection.disconnect(self._close_session)
    
    async def _create_session(self) -> None:
        """Create aiohttp session."""
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            timeout=timeout
        )
        self.logger.debug("HTTP session created")
    
    async def _close_session(self) -> None:
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
        self.logger.debug("HTTP session closed")
    
    async def _authenticate(self) -> None:
        """Verify authentication with Alpaca."""
        account = await self._fetch_account_info()
        if account:
            self.logger.info(f"Authenticated to Alpaca account: {account.get('account_number', 'N/A')}")
        else:
            raise Exception("Failed to authenticate with Alpaca")
    
    async def validate_order(self, order: Order) -> tuple[bool, Optional[str]]:
        """Validate order before submission."""
        # Use composed validator
        return await self.validator.validate(
            order,
            self.connection.connected,
            self._alpaca_specific_validation
        )
    
    async def _alpaca_specific_validation(self, order: Order) -> tuple[bool, Optional[str]]:
        """Alpaca-specific order validation."""
        # Check if market is open for market orders
        if order.order_type == OrderType.MARKET:
            clock = await self._get_market_clock()
            if clock and not clock.get('is_open', False):
                return False, "Market is closed for market orders"
        
        # Check if asset is tradable
        asset = await self._get_asset(order.symbol)
        if not asset:
            return False, f"Asset {order.symbol} not found"
        if not asset.get('tradable', False):
            return False, f"Asset {order.symbol} is not tradable"
        
        return True, None
    
    async def submit_order(self, order: Order) -> str:
        """
        Submit order to Alpaca.
        
        Async I/O operation - submits order and returns immediately.
        Fill notifications come later through get_recent_fills().
        """
        # Build order payload (sync operation)
        payload = self._build_order_payload(order)
        
        # Submit via rate-limited API call (async I/O)
        response = await self.rate_limiter.execute(
            self._submit_order_request,
            payload
        )
        
        # Extract Alpaca order ID
        alpaca_id = response.get('id')
        if not alpaca_id:
            raise Exception(f"No order ID in response: {response}")
        
        # Track order (sync operation)
        self._pending_orders.add(order.order_id)
        self._order_map[order.order_id] = alpaca_id
        self._alpaca_to_internal[alpaca_id] = order.order_id
        
        self.logger.info(f"Order submitted: {order.order_id} -> Alpaca {alpaca_id}")
        return alpaca_id
    
    def _build_order_payload(self, order: Order) -> Dict[str, Any]:
        """Build Alpaca API order payload (sync operation)."""
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": order.order_type.value.lower(),
            "time_in_force": order.time_in_force or "day"
        }
        
        # Add prices based on order type
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price:
                payload["limit_price"] = str(order.price)
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price:
                payload["stop_price"] = str(order.stop_price)
        
        return payload
    
    async def _submit_order_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit order via API (async I/O)."""
        async with self._session.post(
            f"{self.config.base_url}/v2/orders",
            json=payload
        ) as response:
            if response.status == 201:
                return await response.json()
            else:
                text = await response.text()
                raise Exception(f"Order submission failed: {response.status} - {text}")
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order (async I/O)."""
        alpaca_id = self._order_map.get(order_id, order_id)
        
        try:
            await self.rate_limiter.execute(
                self._cancel_order_request,
                alpaca_id
            )
            
            # Update tracking (sync)
            self._pending_orders.discard(order_id)
            
            self.logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Cancel failed for {order_id}: {e}")
            return False
    
    async def _cancel_order_request(self, alpaca_id: str) -> None:
        """Cancel order via API (async I/O)."""
        async with self._session.delete(
            f"{self.config.base_url}/v2/orders/{alpaca_id}"
        ) as response:
            if response.status not in [204, 422]:  # 422 = already done
                text = await response.text()
                raise Exception(f"Cancel failed: {response.status} - {text}")
    
    async def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status (async I/O with caching)."""
        alpaca_id = self._order_map.get(order_id, order_id)
        
        try:
            order_data = await self.rate_limiter.execute(
                self._get_order_request,
                alpaca_id
            )
            return order_data.get('status') if order_data else None
            
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return None
    
    async def _get_order_request(self, alpaca_id: str) -> Dict[str, Any]:
        """Get order via API (async I/O)."""
        async with self._session.get(
            f"{self.config.base_url}/v2/orders/{alpaca_id}"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {}
    
    async def get_recent_fills(self) -> List[Fill]:
        """
        Get recent fills (async I/O).
        
        This is called periodically by the execution engine to check for fills.
        Returns fills for orders we're tracking.
        """
        try:
            # Get filled orders from API
            filled_orders = await self.rate_limiter.execute(
                self._get_filled_orders_request
            )
            
            # Convert to Fill objects (sync operation)
            fills = []
            for order_data in filled_orders:
                fill = self._order_to_fill(order_data)
                if fill and fill.order_id in self._pending_orders:
                    fills.append(fill)
                    self._pending_orders.discard(fill.order_id)
            
            return fills
            
        except Exception as e:
            self.logger.error(f"Failed to get fills: {e}")
            return []
    
    async def _get_filled_orders_request(self) -> List[Dict[str, Any]]:
        """Get filled orders via API (async I/O)."""
        async with self._session.get(
            f"{self.config.base_url}/v2/orders",
            params={
                "status": "filled",
                "limit": 100,
                "direction": "desc"
            }
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return []
    
    def _order_to_fill(self, order_data: Dict[str, Any]) -> Optional[Fill]:
        """Convert Alpaca order to Fill object (sync operation)."""
        if not order_data.get('filled_at'):
            return None
        
        alpaca_id = order_data['id']
        
        # Find our internal order ID
        internal_id = None
        for int_id, alp_id in self._order_map.items():
            if alp_id == alpaca_id:
                internal_id = int_id
                break
        
        if not internal_id:
            return None  # Not our order
        
        return Fill(
            fill_id=f"alpaca_{alpaca_id}",
            order_id=internal_id,
            symbol=order_data['symbol'],
            side=OrderSide(order_data['side']),
            quantity=Decimal(order_data['filled_qty']),
            price=Decimal(order_data['filled_avg_price'] or '0'),
            commission=Decimal('0'),  # Alpaca is commission-free
            executed_at=datetime.fromisoformat(
                order_data['filled_at'].replace('Z', '+00:00')
            ),
            status=FillStatus.FILLED,
            metadata={
                'alpaca_order_id': alpaca_id,
                'order_type': order_data['order_type'],
                'time_in_force': order_data['time_in_force']
            }
        )
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions (async I/O with caching)."""
        # Check cache first (sync)
        cached = self.cache.get_positions()
        if cached is not None:
            return cached
        
        # Fetch from API (async I/O)
        positions = await self.rate_limiter.execute(
            self._fetch_positions
        )
        
        # Update cache (sync)
        self.cache.set_positions(positions)
        
        return positions
    
    async def _fetch_positions(self) -> Dict[str, Position]:
        """Fetch positions via API (async I/O)."""
        async with self._session.get(
            f"{self.config.base_url}/v2/positions"
        ) as response:
            if response.status != 200:
                return {}
            
            positions_data = await response.json()
            
            # Convert to Position objects (sync)
            positions = {}
            for pos_data in positions_data:
                position = self._parse_position(pos_data)
                if position:
                    positions[position.symbol] = position
            
            return positions
    
    def _parse_position(self, pos_data: Dict[str, Any]) -> Optional[Position]:
        """Parse Alpaca position data (sync operation)."""
        try:
            qty = Decimal(pos_data['qty'])
            if qty == 0:
                return None
            
            return Position(
                symbol=pos_data['symbol'],
                quantity=qty,
                avg_price=Decimal(pos_data['avg_entry_price']),
                current_price=Decimal(pos_data['current_price'] or '0'),
                unrealized_pnl=Decimal(pos_data['unrealized_pl'] or '0'),
                realized_pnl=Decimal('0'),  # Not provided by Alpaca
                metadata={
                    'market_value': pos_data['market_value'],
                    'cost_basis': pos_data['cost_basis'],
                    'side': pos_data['side']
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to parse position: {e}")
            return None
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info (async I/O with caching)."""
        # Check cache first (sync)
        cached = self.cache.get_account_info()
        if cached is not None:
            return cached
        
        # Fetch from API (async I/O)
        account = await self.rate_limiter.execute(
            self._fetch_account_info
        )
        
        # Update cache (sync)
        self.cache.set_account_info(account)
        
        return account
    
    async def _fetch_account_info(self) -> Dict[str, Any]:
        """Fetch account info via API (async I/O)."""
        async with self._session.get(
            f"{self.config.base_url}/v2/account"
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {}
    
    async def _get_market_clock(self) -> Optional[Dict[str, Any]]:
        """Get market clock (async I/O)."""
        try:
            async with self._session.get(
                f"{self.config.base_url}/v2/clock"
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            pass
        return None
    
    async def _get_asset(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get asset info (async I/O)."""
        try:
            async with self._session.get(
                f"{self.config.base_url}/v2/assets/{symbol}"
            ) as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            pass
        return None
    
    # Additional methods for order monitoring
    async def get_order_updates(self) -> List[Dict[str, Any]]:
        """Get order status updates for pending orders."""
        updates = []
        
        for internal_id in list(self._pending_orders):
            status = await self.get_order_status(internal_id)
            if status:
                updates.append({
                    'order_id': internal_id,
                    'status': status
                })
                
                # Remove from pending if terminal state
                if status in ['filled', 'cancelled', 'rejected', 'expired']:
                    self._pending_orders.discard(internal_id)
        
        return updates
    
    # WebSocket streaming methods
    
    async def stream_order_updates(self) -> AsyncIterator[TradeUpdate]:
        """
        Stream order updates via WebSocket.
        
        Provides real-time notifications for order events.
        Falls back to polling if WebSocket not available.
        """
        if not self._trade_stream or not self._trade_stream.is_connected:
            self.logger.debug("Trade stream not available")
            return
        
        try:
            async for update in self._trade_stream.stream_updates():
                # Only yield updates for orders we're tracking
                if update.order_id in self._alpaca_to_internal:
                    yield update
                    
        except Exception as e:
            self.logger.error(f"Error streaming order updates: {e}")
    
    def trade_update_to_fill(self, update: TradeUpdate) -> Optional[Fill]:
        """
        Convert trade update to Fill object.
        
        Only converts FILL and PARTIAL_FILL events.
        """
        if update.event not in [TradeUpdateType.FILL, TradeUpdateType.PARTIAL_FILL]:
            return None
        
        # Get internal order ID
        internal_id = self._alpaca_to_internal.get(update.order_id)
        if not internal_id:
            return None
        
        # Create Fill object
        return Fill(
            fill_id=f"alpaca_{update.order_id}_{update.timestamp.timestamp()}",
            order_id=internal_id,
            symbol=update.symbol,
            side=OrderSide(update.side),
            quantity=Decimal(str(update.filled_quantity)),
            price=Decimal(str(update.filled_avg_price or 0)),
            commission=Decimal('0'),  # Alpaca is commission-free
            executed_at=update.timestamp,
            status=FillStatus.FILLED if update.event == TradeUpdateType.FILL else FillStatus.PARTIAL,
            metadata={
                'alpaca_order_id': update.order_id,
                'event_type': update.event.value,
                'total_quantity': update.quantity
            }
        )
    
    @property
    def has_trade_stream(self) -> bool:
        """Check if trade stream is available."""
        return self._trade_stream is not None and self._trade_stream.is_connected


# Factory function
def create_alpaca_broker(
    api_key: str,
    secret_key: str,
    paper_trading: bool = True,
    **kwargs
) -> CleanAlpacaBroker:
    """
    Create a clean async Alpaca broker.
    
    Args:
        api_key: Alpaca API key
        secret_key: Alpaca secret key  
        paper_trading: Use paper trading endpoints
        **kwargs: Additional configuration options
    
    Returns:
        Configured CleanAlpacaBroker instance
    """
    # Determine URLs based on paper trading
    if paper_trading:
        base_url = "https://paper-api.alpaca.markets"
    else:
        base_url = "https://api.alpaca.markets"
    
    config = AlpacaConfig(
        broker_name="alpaca",
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url,
        data_url="https://data.alpaca.markets",
        paper_trading=paper_trading,
        **kwargs
    )
    
    return CleanAlpacaBroker(config)
