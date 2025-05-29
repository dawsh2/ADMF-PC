"""Backtest broker implementation for simulated execution."""

import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime
import uuid
from dataclasses import dataclass, field

from .protocols import (
    Broker, Order, Fill, Position, OrderStatus,
    OrderSide, OrderType, FillType
)
from .market_simulation import MarketSimulator
from ..core.logging.structured import get_logger


logger = get_logger(__name__)


@dataclass
class BacktestAccount:
    """Backtest account state."""
    cash: float
    initial_cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    orders: Dict[str, Order] = field(default_factory=dict)
    order_status: Dict[str, OrderStatus] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    
    @property
    def equity(self) -> float:
        """Calculate total equity."""
        position_value = sum(
            pos.quantity * pos.current_price
            for pos in self.positions.values()
        )
        return self.cash + position_value
    
    @property
    def buying_power(self) -> float:
        """Calculate available buying power."""
        # Simple implementation - could be enhanced with margin
        return self.cash


class BacktestBroker:
    """Backtest broker for simulated order execution."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        market_simulator: Optional[MarketSimulator] = None
    ):
        """Initialize backtest broker."""
        self.account = BacktestAccount(
            cash=initial_cash,
            initial_cash=initial_cash
        )
        self.market_simulator = market_simulator
        self._order_lock = asyncio.Lock()
        self._position_lock = asyncio.Lock()
        
        logger.info(f"Initialized BacktestBroker with {initial_cash} cash")
    
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        async with self._order_lock:
            # Store order
            self.account.orders[order.order_id] = order
            self.account.order_status[order.order_id] = OrderStatus.SUBMITTED
            
            logger.info(
                f"Order submitted: {order.order_id} - "
                f"{order.side.name} {order.quantity} {order.symbol} "
                f"@ {order.order_type.name}"
            )
            
            return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        async with self._order_lock:
            if order_id not in self.account.orders:
                logger.warning(f"Order not found: {order_id}")
                return False
            
            status = self.account.order_status.get(order_id)
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"Cannot cancel order {order_id} with status {status}")
                return False
            
            self.account.order_status[order_id] = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""
        return self.account.order_status.get(order_id, OrderStatus.REJECTED)
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        async with self._position_lock:
            return self.account.positions.copy()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        return {
            "cash": self.account.cash,
            "equity": self.account.equity,
            "buying_power": self.account.buying_power,
            "initial_cash": self.account.initial_cash,
            "position_count": len(self.account.positions),
            "open_orders": sum(
                1 for status in self.account.order_status.values()
                if status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
            ),
            "total_fills": len(self.account.fills)
        }
    
    async def execute_fill(self, fill: Fill) -> bool:
        """Execute fill and update positions."""
        async with self._position_lock:
            order = self.account.orders.get(fill.order_id)
            if not order:
                logger.error(f"Order not found for fill: {fill.order_id}")
                return False
            
            # Update cash
            if order.side == OrderSide.BUY:
                cost = fill.quantity * fill.price + fill.commission
                if cost > self.account.cash:
                    logger.error(f"Insufficient cash for fill: {cost} > {self.account.cash}")
                    return False
                self.account.cash -= cost
            else:  # SELL
                proceeds = fill.quantity * fill.price - fill.commission
                self.account.cash += proceeds
            
            # Update position
            symbol = order.symbol
            if symbol not in self.account.positions:
                if order.side == OrderSide.BUY:
                    self.account.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=fill.quantity,
                        avg_price=fill.price,
                        current_price=fill.price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0
                    )
                else:
                    logger.error(f"Cannot sell {symbol} - no position")
                    return False
            else:
                position = self.account.positions[symbol]
                if order.side == OrderSide.BUY:
                    # Update average price
                    total_cost = (position.quantity * position.avg_price + 
                                 fill.quantity * fill.price)
                    position.quantity += fill.quantity
                    position.avg_price = total_cost / position.quantity
                else:  # SELL
                    if fill.quantity > position.quantity:
                        logger.error(
                            f"Cannot sell {fill.quantity} {symbol} - "
                            f"only have {position.quantity}"
                        )
                        return False
                    
                    # Calculate realized P&L
                    realized_pnl = fill.quantity * (fill.price - position.avg_price)
                    position.realized_pnl += realized_pnl
                    position.quantity -= fill.quantity
                    
                    # Remove position if fully closed
                    if position.quantity == 0:
                        del self.account.positions[symbol]
            
            # Store fill
            self.account.fills.append(fill)
            
            # Update order status
            if fill.fill_type == FillType.FULL:
                self.account.order_status[fill.order_id] = OrderStatus.FILLED
            else:
                self.account.order_status[fill.order_id] = OrderStatus.PARTIAL
            
            logger.info(
                f"Fill executed: {fill.fill_id} - "
                f"{order.side.name} {fill.quantity} {symbol} @ {fill.price}"
            )
            
            return True
    
    async def update_position_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for positions."""
        async with self._position_lock:
            for symbol, price in prices.items():
                if symbol in self.account.positions:
                    position = self.account.positions[symbol]
                    position.current_price = price
                    position.unrealized_pnl = (
                        position.quantity * (price - position.avg_price)
                    )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        total_commission = sum(fill.commission for fill in self.account.fills)
        total_slippage = sum(fill.slippage for fill in self.account.fills)
        
        return {
            "total_fills": len(self.account.fills),
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "avg_commission_per_fill": (
                total_commission / len(self.account.fills)
                if self.account.fills else 0
            ),
            "avg_slippage_per_fill": (
                total_slippage / len(self.account.fills)
                if self.account.fills else 0
            ),
            "current_equity": self.account.equity,
            "total_return": (
                (self.account.equity - self.account.initial_cash) /
                self.account.initial_cash
            )
        }