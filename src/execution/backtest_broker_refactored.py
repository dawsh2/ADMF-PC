"""Refactored backtest broker that uses Risk module's portfolio state as single source of truth."""

import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from .protocols import (
    Broker, Order, Fill, OrderStatus,
    OrderSide, OrderType, FillType, FillStatus
)
from .market_simulation import MarketSimulator
import logging
from ..risk.protocols import PortfolioStateProtocol

logger = logging.getLogger(__name__)


@dataclass
class OrderTracker:
    """Tracks orders without duplicating position state."""
    orders: Dict[str, Order] = field(default_factory=dict)
    order_status: Dict[str, OrderStatus] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)


class BacktestBrokerRefactored:
    """
    Refactored backtest broker that delegates position tracking to Risk module.
    
    This broker:
    - Does NOT maintain its own position state
    - Uses PortfolioStateProtocol as the single source of truth
    - Only tracks orders and fills
    - Validates orders against the portfolio state
    """
    
    def __init__(
        self,
        initial_cash: Optional[Decimal] = None,
        risk_portfolio_container: Optional[Any] = None,
        portfolio_state: Optional[PortfolioStateProtocol] = None,
        market_simulator: Optional[MarketSimulator] = None
    ):
        """Initialize backtest broker with reference to portfolio state.
        
        Args:
            initial_cash: Initial cash (used if risk_portfolio_container provided)
            risk_portfolio_container: Risk portfolio container (alternative to portfolio_state)
            portfolio_state: The authoritative portfolio state from Risk module
            market_simulator: Market simulator for realistic fills
        """
        # Handle different initialization patterns
        if risk_portfolio_container is not None:
            self.portfolio_state = risk_portfolio_container.get_portfolio_state()
        elif portfolio_state is not None:
            self.portfolio_state = portfolio_state
        else:
            raise ValueError("Either risk_portfolio_container or portfolio_state must be provided")
        
        self.market_simulator = market_simulator
        self.order_tracker = OrderTracker()
        self._order_lock = asyncio.Lock()
        
        logger.info(
            f"Initialized BacktestBroker with portfolio state, initial_capital={self.portfolio_state.get_cash_balance()}"
        )
    
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        async with self._order_lock:
            # Validate order against portfolio state
            if not await self._validate_order(order):
                logger.warning(
                    f"Order validation failed - ID: {order.order_id}, reason: Insufficient funds or position"
                )
                self.order_tracker.order_status[order.order_id] = OrderStatus.REJECTED
                return order.order_id
            
            # Store order
            self.order_tracker.orders[order.order_id] = order
            self.order_tracker.order_status[order.order_id] = OrderStatus.SUBMITTED
            
            logger.info(
                f"Order submitted - ID: {order.order_id}, side: {order.side.name}, "
                f"quantity: {order.quantity}, symbol: {order.symbol}, type: {order.order_type.name}"
            )
            
            return order.order_id
    
    async def _validate_order(self, order: Order) -> bool:
        """Validate order against portfolio state."""
        # Convert to Decimal for consistency
        quantity = Decimal(str(order.quantity))
        
        if order.side == OrderSide.BUY:
            # Check buying power
            price = Decimal(str(order.price if order.price else 0))
            if price > 0:
                required_cash = quantity * price
                available_cash = self.portfolio_state.get_cash_balance()
                return required_cash <= available_cash
            # For market orders, we'll check during execution
            return True
        else:  # SELL
            # Check position
            position = self.portfolio_state.get_position(order.symbol)
            if not position:
                return False
            return quantity <= abs(position.quantity)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        async with self._order_lock:
            if order_id not in self.order_tracker.orders:
                logger.warning(f"Order not found for cancellation: {order_id}")
                return False
            
            status = self.order_tracker.order_status.get(order_id)
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"Cannot cancel order {order_id} with status {status.name}")
                return False
            
            self.order_tracker.order_status[order_id] = OrderStatus.CANCELLED
            logger.info(f"Order cancelled - ID: {order_id}")
            return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""
        return self.order_tracker.order_status.get(order_id, OrderStatus.REJECTED)
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions from portfolio state.
        
        Returns positions in a format compatible with execution module.
        """
        positions = self.portfolio_state.get_all_positions()
        
        # Convert to execution module format
        return {
            symbol: {
                "symbol": pos.symbol,
                "quantity": float(pos.quantity),
                "avg_price": float(pos.average_price),
                "current_price": float(pos.current_price),
                "unrealized_pnl": float(pos.unrealized_pnl),
                "realized_pnl": float(pos.realized_pnl)
            }
            for symbol, pos in positions.items()
        }
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from portfolio state."""
        metrics = self.portfolio_state.get_risk_metrics()
        positions = self.portfolio_state.get_all_positions()
        
        return {
            "cash": float(metrics.cash_balance),
            "equity": float(metrics.total_value),
            "buying_power": float(metrics.cash_balance),  # Simple implementation
            "positions_value": float(metrics.positions_value),
            "position_count": len(positions),
            "open_orders": sum(
                1 for status in self.order_tracker.order_status.values()
                if status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]
            ),
            "total_fills": len(self.order_tracker.fills),
            "unrealized_pnl": float(metrics.unrealized_pnl),
            "realized_pnl": float(metrics.realized_pnl)
        }
    
    async def execute_fill(self, fill: Fill) -> bool:
        """Execute fill - just records it, portfolio state handles position updates.
        
        The fill should be processed by Risk module's portfolio state.
        This method just tracks the fill for execution history.
        """
        async with self._order_lock:
            order = self.order_tracker.orders.get(fill.order_id)
            if not order:
                logger.error(f"Order not found for fill: {fill.order_id}")
                return False
            
            # Store fill for history
            self.order_tracker.fills.append(fill)
            
            # Update order status
            if fill.fill_type == FillType.FULL:
                self.order_tracker.order_status[fill.order_id] = OrderStatus.FILLED
            else:
                self.order_tracker.order_status[fill.order_id] = OrderStatus.PARTIAL
            
            logger.info(
                "fill_recorded",
                fill_id=fill.fill_id,
                order_id=fill.order_id,
                side=order.side.name,
                quantity=fill.quantity,
                symbol=order.symbol,
                price=fill.price
            )
            
            # Note: The actual position update should be done by calling
            # portfolio_state.update_position() from the component that
            # orchestrates the execution flow
            
            return True
    
    def create_fill_for_portfolio_update(self, fill: Fill) -> Dict[str, Any]:
        """Convert Fill to format expected by portfolio state.
        
        Returns:
            Dictionary with fill data for portfolio update
        """
        order = self.order_tracker.orders.get(fill.order_id)
        if not order:
            raise ValueError(f"Order {fill.order_id} not found")
        
        return {
            "order_id": fill.order_id,
            "symbol": fill.symbol,
            "side": "buy" if fill.side == OrderSide.BUY else "sell",
            "quantity": Decimal(str(fill.quantity)),
            "price": Decimal(str(fill.price)),
            "timestamp": fill.executed_at,
            "commission": Decimal(str(fill.commission))
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        fills = self.order_tracker.fills
        
        total_commission = sum(fill.commission for fill in fills)
        total_slippage = sum(fill.slippage for fill in fills)
        
        # Get portfolio metrics for return calculation
        metrics = self.portfolio_state.get_risk_metrics()
        
        return {
            "total_fills": len(fills),
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "avg_commission_per_fill": (
                total_commission / len(fills) if fills else 0
            ),
            "avg_slippage_per_fill": (
                total_slippage / len(fills) if fills else 0
            ),
            "current_equity": float(metrics.total_value),
            "total_return": float(
                (metrics.total_value - metrics.cash_balance) / metrics.cash_balance
                if metrics.cash_balance > 0 else 0
            )
        }
    
    async def process_pending_orders(self, market_data: Dict[str, float]) -> List[Fill]:
        """Process pending orders with current market data.
        
        This method simulates order execution based on market data.
        Returns list of fills that need to be processed by portfolio state.
        """
        fills_to_process = []
        
        async with self._order_lock:
            for order_id, order in self.order_tracker.orders.items():
                status = self.order_tracker.order_status.get(order_id)
                
                # Skip non-pending orders
                if status not in [OrderStatus.SUBMITTED, OrderStatus.PENDING]:
                    continue
                
                # Get market price
                market_price = market_data.get(order.symbol)
                if not market_price:
                    continue
                
                # Check if order should fill
                should_fill = False
                fill_price = market_price
                
                if order.order_type == OrderType.MARKET:
                    should_fill = True
                elif order.order_type == OrderType.LIMIT:
                    if order.side == OrderSide.BUY and market_price <= order.price:
                        should_fill = True
                        fill_price = order.price
                    elif order.side == OrderSide.SELL and market_price >= order.price:
                        should_fill = True
                        fill_price = order.price
                
                if should_fill:
                    # Create fill
                    fill = Fill(
                        fill_id=f"FILL-{uuid.uuid4().hex[:8]}",
                        order_id=order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=fill_price,
                        commission=self._calculate_commission(order.quantity, fill_price),
                        slippage=0.0,  # Could be enhanced
                        fill_type=FillType.FULL,
                        executed_at=datetime.now()
                    )
                    
                    # Execute fill (just records it)
                    await self.execute_fill(fill)
                    fills_to_process.append(fill)
        
        return fills_to_process
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade."""
        # Simple fixed + per share model
        return 1.0 + (0.005 * quantity)  # $1 + $0.005/share    
    def simulate_fill(self, order: Order, market_price: Decimal) -> Optional[Fill]:
        """Simulate fill execution for an order.
        
        Args:
            order: Order to fill
            market_price: Current market price
            
        Returns:
            Fill object if successful, None otherwise
        """
        # Simple fill simulation - in production would use MarketSimulator
        if order.order_id not in self.order_tracker.orders:
            logger.warning(f"Order {order.order_id} not found")
            return None
        
        # Calculate fill price based on order type
        fill_price = market_price
        if order.order_type == OrderType.LIMIT and order.price:
            # For limit orders, check if market price allows execution
            if order.side == OrderSide.BUY and market_price > order.price:
                return None  # Market price too high for buy limit
            elif order.side == OrderSide.SELL and market_price < order.price:
                return None  # Market price too low for sell limit
            fill_price = Decimal(str(order.price))
        
        # Simple slippage calculation
        slippage = market_price * Decimal("0.0001")  # 0.01% slippage
        if order.side == OrderSide.BUY:
            fill_price += slippage
        else:
            fill_price -= slippage
        
        # Calculate commission
        commission = max(order.quantity * fill_price * Decimal("0.001"), Decimal("1.0"))  # 0.1% or $1 minimum
        
        # Create fill
        fill = Fill(
            fill_id=f"FILL-{uuid.uuid4().hex[:8]}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=float(fill_price),
            commission=float(commission),
            slippage=float(slippage),
            fill_type=FillType.FULL,
            status=FillStatus.FILLED,
            executed_at=datetime.now(),
            metadata={
                "market_price": float(market_price),
                "simulated": True
            }
        )
        
        # Update order status
        self.order_tracker.order_status[order.order_id] = OrderStatus.FILLED
        self.order_tracker.fills.append(fill)
        
        logger.info(
            f"Simulated fill: {order.side.name} {fill.quantity} {order.symbol} "
            f"@ {fill.price:.2f} (market: {market_price:.2f}, "
            f"slippage: {slippage:.4f}, commission: {commission:.2f})"
        )
        
        return fill