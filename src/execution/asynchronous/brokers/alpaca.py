"""
Alpaca broker implementation for live trading.

Real Alpaca API integration with proper async patterns.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal

from ...types import Order, Fill, Position, OrderSide, OrderType, FillStatus
from .base import BaseLiveBroker


class AlpacaBroker(BaseLiveBroker):
    """
    Alpaca broker implementation for live trading.
    
    Integrates with Alpaca's REST API and WebSocket streams
    for real-time trading operations.
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper_trading: bool = True
    ):
        # Alpaca URLs
        if paper_trading:
            base_url = "https://paper-api.alpaca.markets"
            data_url = "https://data.alpaca.markets"
        else:
            base_url = "https://api.alpaca.markets"
            data_url = "https://data.alpaca.markets"
        
        super().__init__(
            broker_name="alpaca",
            api_key=api_key,
            secret_key=secret_key,
            base_url=base_url,
            paper_trading=paper_trading
        )
        
        self.data_url = data_url
        
        # Alpaca-specific settings
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json"
        }
        
        # WebSocket for real-time data (placeholder)
        self._ws_session = None
        
        # Order ID mapping
        self._internal_to_alpaca: Dict[str, str] = {}
        self._alpaca_to_internal: Dict[str, str] = {}
    
    @property
    def supported_order_types(self) -> List[str]:
        """Alpaca supported order types."""
        return ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
    
    @property
    def min_order_size(self) -> float:
        """Alpaca minimum order size."""
        return 1.0  # 1 share minimum
    
    async def _establish_connection(self) -> None:
        """Establish connection to Alpaca API."""
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(
            headers=self._headers,
            timeout=timeout
        )
    
    async def _authenticate(self) -> None:
        """Authenticate with Alpaca API."""
        try:
            # Test authentication by getting account info
            async with self._session.get(f"{self.base_url}/v2/account") as response:
                if response.status == 200:
                    self.logger.info("Alpaca authentication successful")
                    return
                else:
                    error_text = await response.text()
                    raise Exception(f"Authentication failed: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Alpaca authentication error: {e}")
    
    async def _close_connection(self) -> None:
        """Close connection to Alpaca API."""
        if self._session:
            await self._session.close()
            self._session = None
        
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None
    
    async def _validate_order_with_broker(self, order: Order) -> tuple[bool, Optional[str]]:
        """Alpaca-specific order validation."""
        try:
            # Check market hours for market orders
            if order.order_type == OrderType.MARKET:
                # Get market status
                async with self._session.get(f"{self.base_url}/v2/clock") as response:
                    if response.status == 200:
                        clock_data = await response.json()
                        if not clock_data.get('is_open', False):
                            return False, "Market is closed for market orders"
            
            # Check asset tradability
            async with self._session.get(f"{self.base_url}/v2/assets/{order.symbol}") as response:
                if response.status == 200:
                    asset_data = await response.json()
                    if not asset_data.get('tradable', False):
                        return False, f"Asset {order.symbol} is not tradable"
                else:
                    return False, f"Asset {order.symbol} not found"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    async def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca."""
        if not self._connected:
            raise Exception("Not connected to Alpaca")
        
        # Prepare order data
        order_data = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": order.order_type.value.lower(),
            "time_in_force": order.time_in_force or "day"
        }
        
        # Add price for limit orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price:
            order_data["limit_price"] = str(order.price)
        
        # Add stop price for stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price:
            order_data["stop_price"] = str(order.stop_price)
        
        try:
            async with self._session.post(
                f"{self.base_url}/v2/orders",
                json=order_data
            ) as response:
                
                if response.status == 201:
                    alpaca_order = await response.json()
                    alpaca_order_id = alpaca_order["id"]
                    
                    # Store ID mapping
                    self._internal_to_alpaca[order.order_id] = alpaca_order_id
                    self._alpaca_to_internal[alpaca_order_id] = order.order_id
                    
                    self.logger.debug(f"Order submitted to Alpaca: {order.order_id} -> {alpaca_order_id}")
                    return alpaca_order_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Order submission failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self.logger.error(f"Alpaca order submission error: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in Alpaca."""
        if not self._connected:
            return False
        
        # Get Alpaca order ID
        alpaca_order_id = self._internal_to_alpaca.get(order_id, order_id)
        
        try:
            async with self._session.delete(
                f"{self.base_url}/v2/orders/{alpaca_order_id}"
            ) as response:
                
                if response.status == 204:
                    self.logger.debug(f"Order cancelled in Alpaca: {alpaca_order_id}")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.warning(f"Order cancellation failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Alpaca order cancellation error: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status from Alpaca."""
        if not self._connected:
            return None
        
        # Get Alpaca order ID
        alpaca_order_id = self._internal_to_alpaca.get(order_id, order_id)
        
        try:
            async with self._session.get(
                f"{self.base_url}/v2/orders/{alpaca_order_id}"
            ) as response:
                
                if response.status == 200:
                    order_data = await response.json()
                    return order_data.get("status")
                else:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    async def _fetch_positions(self) -> Dict[str, Position]:
        """Fetch positions from Alpaca."""
        if not self._connected:
            return {}
        
        try:
            async with self._session.get(f"{self.base_url}/v2/positions") as response:
                if response.status == 200:
                    positions_data = await response.json()
                    positions = {}
                    
                    for pos_data in positions_data:
                        symbol = pos_data["symbol"]
                        position = Position(
                            symbol=symbol,
                            quantity=Decimal(pos_data["qty"]),
                            avg_price=Decimal(pos_data["avg_entry_price"]),
                            current_price=Decimal(pos_data["market_value"]) / Decimal(pos_data["qty"]) 
                                         if float(pos_data["qty"]) != 0 else Decimal("0"),
                            unrealized_pnl=Decimal(pos_data["unrealized_pl"]),
                            realized_pnl=Decimal("0"),  # Alpaca doesn't provide this directly
                            metadata={
                                "market_value": pos_data["market_value"],
                                "cost_basis": pos_data["cost_basis"],
                                "side": pos_data["side"]
                            }
                        )
                        positions[symbol] = position
                    
                    return positions
                else:
                    self.logger.error(f"Error fetching positions: {response.status}")
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return {}
    
    async def _fetch_account_info(self) -> Dict[str, Any]:
        """Fetch account information from Alpaca."""
        if not self._connected:
            return {}
        
        try:
            async with self._session.get(f"{self.base_url}/v2/account") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Error fetching account info: {response.status}")
                    return {}
                    
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")
            return {}
    
    async def get_recent_fills(self) -> List[Fill]:
        """Get recent fills from Alpaca."""
        if not self._connected:
            return []
        
        try:
            # Get recent orders with fills
            async with self._session.get(
                f"{self.base_url}/v2/orders",
                params={"status": "filled", "limit": 100}
            ) as response:
                
                if response.status == 200:
                    orders_data = await response.json()
                    fills = []
                    
                    for order_data in orders_data:
                        alpaca_order_id = order_data["id"]
                        internal_order_id = self._alpaca_to_internal.get(
                            alpaca_order_id, alpaca_order_id
                        )
                        
                        # Create fill from order data
                        fill = Fill(
                            fill_id=f"alpaca_fill_{alpaca_order_id}",
                            order_id=internal_order_id,
                            symbol=order_data["symbol"],
                            side=OrderSide(order_data["side"]),
                            quantity=Decimal(order_data["filled_qty"]),
                            price=Decimal(order_data["filled_avg_price"] or "0"),
                            commission=Decimal("0"),  # Alpaca is commission-free
                            executed_at=datetime.fromisoformat(
                                order_data["filled_at"].replace("Z", "+00:00")
                            ),
                            status=FillStatus.FILLED,
                            metadata={
                                "alpaca_order_id": alpaca_order_id,
                                "submitted_at": order_data["submitted_at"],
                                "time_in_force": order_data["time_in_force"]
                            }
                        )
                        fills.append(fill)
                    
                    return fills
                else:
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error getting recent fills: {e}")
            return []