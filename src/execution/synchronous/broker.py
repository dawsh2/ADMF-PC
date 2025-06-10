"""
Synchronous simulated broker for backtesting.

High-performance order simulation without async overhead.
"""

import logging
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime

from ..types import Order, Fill, Position, OrderSide, FillStatus
from ..sync_protocols import SlippageModel, CommissionModel, LiquidityModel
from .models import (
    ZeroSlippageModel, 
    ZeroCommissionModel, 
    UnlimitedLiquidityModel
)

logger = logging.getLogger(__name__)


class SimulatedBroker:
    """
    Synchronous simulated broker for backtesting.
    
    Provides order execution simulation using configurable
    market models for slippage, commission, and liquidity.
    """
    
    def __init__(
        self,
        broker_id: str = "simulated",
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        liquidity_model: Optional[LiquidityModel] = None,
        portfolio_state = None
    ):
        self.broker_id = broker_id
        self.slippage_model = slippage_model or ZeroSlippageModel()
        self.commission_model = commission_model or ZeroCommissionModel()
        self.liquidity_model = liquidity_model or UnlimitedLiquidityModel()
        self.portfolio_state = portfolio_state
        
        self.logger = logger.getChild(broker_id)
        
        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._fills: List[Fill] = []
        self._fill_counter = 0
        
        # Market data cache
        self._market_data: Dict[str, Any] = {}
        
        # Supported features
        self._supported_order_types = ['MARKET', 'LIMIT']
        self._min_order_size = 1.0
    
    @property
    def supported_order_types(self) -> List[str]:
        """Get supported order types."""
        return self._supported_order_types.copy()
    
    @property
    def min_order_size(self) -> float:
        """Get minimum order size."""
        return self._min_order_size
    
    def validate_order(self, order: Order) -> tuple[bool, Optional[str]]:
        """Validate order constraints."""
        # Basic validation
        if float(order.quantity) < self.min_order_size:
            return False, f"Order quantity {order.quantity} below minimum {self.min_order_size}"
        
        if order.order_type.value.upper() not in self.supported_order_types:
            return False, f"Order type {order.order_type} not supported"
        
        # Portfolio validation
        if self.portfolio_state:
            # Check buying power for buy orders
            if order.side == OrderSide.BUY:
                market_price = self._get_market_price(order.symbol)
                if market_price:
                    order_value = float(order.quantity) * market_price
                    buying_power = self.portfolio_state.get_buying_power()
                    
                    if order_value > buying_power:
                        return False, f"Insufficient buying power: need {order_value}, have {buying_power}"
            
            # Check position for sell orders
            elif order.side == OrderSide.SELL:
                position = self.portfolio_state.get_position(order.symbol)
                if not position or float(position.quantity) < float(order.quantity):
                    return False, f"Insufficient position to sell: need {order.quantity}"
        
        return True, None
    
    def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        # Store order
        self._orders[order.order_id] = order
        
        self.logger.debug(f"Order submitted: {order.order_id} - {order.side.value} {order.quantity} {order.symbol}")
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id in self._orders:
            # In simulation, we can cancel any non-filled order
            order = self._orders[order_id]
            
            # Check if already filled
            for fill in self._fills:
                if fill.order_id == order_id:
                    self.logger.warning(f"Cannot cancel filled order: {order_id}")
                    return False
            
            # Remove from pending orders
            del self._orders[order_id]
            self.logger.debug(f"Order cancelled: {order_id}")
            return True
        
        return False
    
    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get current order status."""
        # Check if filled
        for fill in self._fills:
            if fill.order_id == order_id:
                return "FILLED"
        
        # Check if pending
        if order_id in self._orders:
            return "PENDING"
        
        return None
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        if self.portfolio_state:
            return self.portfolio_state.get_all_positions()
        return {}
    
    def process_market_data(self, market_data: Dict[str, Any]) -> List[Fill]:
        """Process market data and generate fills."""
        # Update market data cache
        self._market_data.update(market_data)
        
        new_fills = []
        orders_to_remove = []
        
        # Try to execute pending orders
        for order_id, order in self._orders.items():
            # Skip if already filled
            if any(fill.order_id == order_id for fill in self._fills):
                continue
            
            fill = self._try_execute_order(order, market_data)
            if fill:
                new_fills.append(fill)
                self._fills.append(fill)
                orders_to_remove.append(order_id)
        
        # Remove filled orders
        for order_id in orders_to_remove:
            del self._orders[order_id]
        
        return new_fills
    
    def _try_execute_order(self, order: Order, market_data: Dict[str, Any]) -> Optional[Fill]:
        """Try to execute an order against market data."""
        # Get market price for symbol
        market_price = self._get_market_price_from_data(order.symbol, market_data)
        if not market_price:
            return None
        
        # Check liquidity
        can_fill, fill_ratio = self.liquidity_model.can_fill_order(order, market_data)
        if not can_fill:
            return None
        
        # Calculate fill quantity
        fill_quantity = order.quantity * Decimal(str(fill_ratio))
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(
            order, market_price, market_data.get('volume', 0)
        )
        
        # Calculate fill price
        if order.side == OrderSide.BUY:
            fill_price = market_price + slippage
        else:
            fill_price = market_price - slippage
        
        # Calculate commission
        commission = self.commission_model.calculate_commission(order, fill_price)
        
        # Create fill
        self._fill_counter += 1
        fill = Fill(
            fill_id=f"{self.broker_id}_fill_{self._fill_counter}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=Decimal(str(fill_price)),
            commission=Decimal(str(commission)),
            executed_at=datetime.now(),
            status=FillStatus.FILLED if fill_ratio >= 1.0 else FillStatus.PARTIAL
        )
        
        # Update portfolio state
        if self.portfolio_state:
            self.portfolio_state.process_fill(fill)
        
        self.logger.debug(
            f"Order executed: {order.order_id} -> {fill.fill_id} "
            f"({fill.quantity} @ {fill.price}, commission: {fill.commission})"
        )
        
        return fill
    
    def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        return self._get_market_price_from_data(symbol, self._market_data)
    
    def _get_market_price_from_data(self, symbol: str, market_data: Dict[str, Any]) -> Optional[float]:
        """Get market price from market data."""
        # Try different price fields
        for price_field in ['close', 'price', 'last', 'mid']:
            if price_field in market_data:
                price_data = market_data[price_field]
                
                # Handle different data structures
                if isinstance(price_data, dict) and symbol in price_data:
                    return float(price_data[symbol])
                elif isinstance(price_data, (int, float)):
                    return float(price_data)
        
        # Check if symbol-specific data exists
        if symbol in market_data:
            symbol_data = market_data[symbol]
            if isinstance(symbol_data, dict):
                for price_field in ['close', 'price', 'last']:
                    if price_field in symbol_data:
                        return float(symbol_data[price_field])
            elif isinstance(symbol_data, (int, float)):
                return float(symbol_data)
        
        return None
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data cache."""
        return self._market_data.copy()
    
    def clear_market_data(self) -> None:
        """Clear market data cache."""
        self._market_data.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        return {
            'orders_submitted': len(self._orders) + len(self._fills),
            'orders_pending': len(self._orders),
            'fills_generated': len(self._fills),
            'total_commission': sum(float(fill.commission) for fill in self._fills)
        }