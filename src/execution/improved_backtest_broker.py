"""
Improved backtest broker that delegates to Risk module's portfolio state.

This broker eliminates state duplication by using the Risk module's
PortfolioState as the single source of truth for positions and portfolio data.
"""

import asyncio
from typing import Dict, Optional, Any, List
from datetime import datetime
from decimal import Decimal
import uuid
import logging

from ..core.components.protocols import Component, Lifecycle
from .protocols import (
    Broker, Order, Fill, Position, OrderStatus,
    OrderSide, OrderType, FillType, FillStatus
)
from ..risk.protocols import PortfolioStateProtocol

logger = logging.getLogger(__name__)


class BacktestBrokerRefactored(Component, Lifecycle):
    """
    Improved backtest broker using Risk module's portfolio state.
    
    This broker delegates all position and portfolio management to the
    Risk module's PortfolioState, ensuring single source of truth and
    eliminating state duplication issues.
    """
    
    def __init__(
        self,
        component_id: str,
        portfolio_state: PortfolioStateProtocol,
        commission_rate: Decimal = Decimal("0.001"),
        slippage_rate: Decimal = Decimal("0.0005")
    ):
        """Initialize broker with portfolio state dependency injection.
        
        Args:
            component_id: Unique component identifier
            portfolio_state: Portfolio state from Risk module (injected)
            commission_rate: Commission rate as decimal (0.001 = 0.1%)
            slippage_rate: Slippage rate as decimal (0.0005 = 0.05%)
        """
        self._component_id = component_id
        self._portfolio_state = portfolio_state
        self._commission_rate = commission_rate
        self._slippage_rate = slippage_rate
        
        # Order tracking (broker's responsibility)
        self._pending_orders: Dict[str, Order] = {}
        self._order_status: Dict[str, OrderStatus] = {}
        self._order_lock = asyncio.Lock()
        
        # Execution tracking (broker's responsibility)
        self._fills: List[Fill] = []
        self._execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_commission': Decimal('0'),
            'total_slippage': Decimal('0')
        }
        
        # Market data for execution
        self._market_prices: Dict[str, Decimal] = {}
        
        # Lifecycle state
        self._initialized = False
        self._running = False
        
        logger.info(f"BacktestBroker initialized - ID: {component_id}")
    
    @property
    def component_id(self) -> str:
        """Component identifier."""
        return self._component_id
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the broker."""
        self._initialized = True
        logger.info(f"BacktestBroker initialized - ID: {self._component_id}")
    
    def start(self) -> None:
        """Start the broker."""
        if not self._initialized:
            raise RuntimeError("Broker not initialized")
        
        self._running = True
        logger.info(f"BacktestBroker started - ID: {self._component_id}")
    
    def stop(self) -> None:
        """Stop the broker."""
        self._running = False
        
        # Cancel all pending orders
        asyncio.create_task(self._cancel_all_pending_orders())
        
        logger.info(f"BacktestBroker stopped - ID: {self._component_id}")
    
    def reset(self) -> None:
        """Reset broker state."""
        # Clear order tracking
        self._pending_orders.clear()
        self._order_status.clear()
        self._fills.clear()
        
        # Reset execution stats
        self._execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_commission': Decimal('0'),
            'total_slippage': Decimal('0')
        }
        
        # Clear market data
        self._market_prices.clear()
        
        logger.info(f"BacktestBroker reset - ID: {self._component_id}")
    
    def teardown(self) -> None:
        """Teardown the broker."""
        # Stop if running
        if self._running:
            self.stop()
        
        # Clear all state
        self.reset()
        
        logger.info(f"BacktestBroker torn down - ID: {self._component_id}")
    
    async def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        if not self._running:
            raise RuntimeError("Broker not running")
        
        async with self._order_lock:
            # Validate order
            validation_result = await self._validate_order(order)
            if not validation_result.is_valid:
                self._order_status[order.order_id] = OrderStatus.REJECTED
                self._execution_stats['rejected_orders'] += 1
                logger.warning(
                    f"Order rejected - ID: {order.order_id}, Reason: {validation_result.reason}"
                )
                raise ValueError(f"Order validation failed: {validation_result.reason}")
            
            # Store order
            self._pending_orders[order.order_id] = order
            self._order_status[order.order_id] = OrderStatus.SUBMITTED
            self._execution_stats['total_orders'] += 1
            
            logger.info(
                f"Order submitted - ID: {order.order_id}, "
                f"{order.side.name} {order.quantity} {order.symbol} @ {order.order_type.name}"
            )
            
            return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        async with self._order_lock:
            if order_id not in self._pending_orders:
                logger.warning(f"Order not found for cancellation: {order_id}")
                return False
            
            status = self._order_status.get(order_id)
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"Cannot cancel order {order_id} with status {status}")
                return False
            
            # Update status
            self._order_status[order_id] = OrderStatus.CANCELLED
            self._execution_stats['cancelled_orders'] += 1
            
            # Remove from pending
            del self._pending_orders[order_id]
            
            logger.info(f"Order cancelled - ID: {order_id}")
            return True
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get current order status."""
        return self._order_status.get(order_id, OrderStatus.REJECTED)
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions from portfolio state."""
        # Delegate to Risk module's portfolio state
        risk_positions = self._portfolio_state.get_all_positions()
        
        # Convert to execution Position format
        execution_positions = {}
        for symbol, risk_pos in risk_positions.items():
            execution_positions[symbol] = Position(
                symbol=symbol,
                quantity=float(risk_pos.quantity),
                avg_price=float(risk_pos.average_price),
                current_price=float(risk_pos.current_price),
                unrealized_pnl=float(risk_pos.unrealized_pnl),
                realized_pnl=float(risk_pos.realized_pnl),
                metadata={
                    'opened_at': risk_pos.opened_at.isoformat(),
                    'last_updated': risk_pos.last_updated.isoformat()
                }
            )
        
        return execution_positions
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information from portfolio state."""
        # Delegate to Risk module's portfolio state
        risk_metrics = self._portfolio_state.get_risk_metrics()
        
        return {
            "cash": float(self._portfolio_state.get_cash_balance()),
            "equity": float(risk_metrics.total_value),
            "positions_value": float(risk_metrics.positions_value),
            "unrealized_pnl": float(risk_metrics.unrealized_pnl),
            "realized_pnl": float(risk_metrics.realized_pnl),
            "leverage": float(risk_metrics.leverage),
            "position_count": len(self._portfolio_state.get_all_positions()),
            "pending_orders": len(self._pending_orders),
            "total_fills": len(self._fills)
        }
    
    async def process_pending_orders(self, market_data: Dict[str, Any]) -> List[Fill]:
        """Process pending orders with current market data."""
        if not self._running:
            return []
        
        # Update market prices
        await self._update_market_data(market_data)
        
        fills = []
        orders_to_remove = []
        
        async with self._order_lock:
            for order_id, order in self._pending_orders.items():
                try:
                    # Attempt to fill order
                    fill = await self._attempt_fill(order)
                    
                    if fill:
                        fills.append(fill)
                        self._fills.append(fill)
                        
                        # Update order status
                        if fill.fill_type == FillType.FULL:
                            self._order_status[order_id] = OrderStatus.FILLED
                            self._execution_stats['filled_orders'] += 1
                            orders_to_remove.append(order_id)
                        else:
                            self._order_status[order_id] = OrderStatus.PARTIAL
                        
                        # Update execution stats
                        self._execution_stats['total_commission'] += Decimal(str(fill.commission))
                        self._execution_stats['total_slippage'] += Decimal(str(fill.slippage))
                        
                        logger.info(
                            f"Order filled - ID: {order_id}, "
                            f"Fill: {fill.quantity} @ {fill.price}, "
                            f"Commission: {fill.commission}, Slippage: {fill.slippage}"
                        )
                        
                        # Update portfolio state through Risk module
                        await self._update_portfolio_state(fill)
                
                except Exception as e:
                    logger.error(f"Error processing order {order_id}: {e}")
                    # Keep order pending for retry
            
            # Remove filled orders
            for order_id in orders_to_remove:
                del self._pending_orders[order_id]
        
        return fills
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        total_orders = self._execution_stats['total_orders']
        fill_rate = (
            self._execution_stats['filled_orders'] / total_orders
            if total_orders > 0 else 0
        )
        
        avg_commission = (
            self._execution_stats['total_commission'] / len(self._fills)
            if self._fills else Decimal('0')
        )
        
        avg_slippage = (
            self._execution_stats['total_slippage'] / len(self._fills)
            if self._fills else Decimal('0')
        )
        
        return {
            **self._execution_stats,
            'fill_rate': float(fill_rate),
            'avg_commission_per_fill': float(avg_commission),
            'avg_slippage_per_fill': float(avg_slippage),
            'total_fills': len(self._fills)
        }
    
    # Private methods
    
    async def _validate_order(self, order: Order) -> 'ValidationResult':
        """Validate order before submission."""
        # Basic validation
        if order.quantity <= 0:
            return ValidationResult(False, "Invalid quantity")
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                return ValidationResult(False, "Invalid limit price")
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                return ValidationResult(False, "Invalid stop price")
        
        # Check buying power for buy orders
        if order.side == OrderSide.BUY:
            estimated_cost = self._estimate_order_cost(order)
            cash_balance = self._portfolio_state.get_cash_balance()
            
            if estimated_cost > cash_balance:
                return ValidationResult(False, f"Insufficient funds: need {estimated_cost}, have {cash_balance}")
        
        # Check position for sell orders
        elif order.side == OrderSide.SELL:
            position = self._portfolio_state.get_position(order.symbol)
            if not position or position.quantity < Decimal(str(order.quantity)):
                available = position.quantity if position else Decimal('0')
                return ValidationResult(False, f"Insufficient position: need {order.quantity}, have {available}")
        
        return ValidationResult(True, "Valid")
    
    def _estimate_order_cost(self, order: Order) -> Decimal:
        """Estimate total cost of order including commission and slippage."""
        # Get market price
        market_price = self._market_prices.get(order.symbol, Decimal('100'))  # Default
        
        # Use limit price if available and better
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price:
            if order.side == OrderSide.BUY:
                execution_price = min(Decimal(str(order.price)), market_price)
            else:
                execution_price = max(Decimal(str(order.price)), market_price)
        else:
            execution_price = market_price
        
        # Add slippage
        if order.side == OrderSide.BUY:
            execution_price *= (Decimal('1') + self._slippage_rate)
        else:
            execution_price *= (Decimal('1') - self._slippage_rate)
        
        # Calculate cost
        gross_cost = Decimal(str(order.quantity)) * execution_price
        commission = gross_cost * self._commission_rate
        
        return gross_cost + commission
    
    async def _attempt_fill(self, order: Order) -> Optional[Fill]:
        """Attempt to fill an order."""
        # Get market price
        market_price = self._market_prices.get(order.symbol)
        if not market_price:
            return None  # No market data available
        
        # Determine if order can be filled
        can_fill, fill_price = self._can_fill_order(order, market_price)
        if not can_fill:
            return None
        
        # Calculate slippage and commission
        slippage = self._calculate_slippage(order, market_price, fill_price)
        commission = self._calculate_commission(order, fill_price)
        
        # Create fill
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,  # Full fill for simplicity
            price=float(fill_price),
            commission=float(commission),
            slippage=float(slippage),
            fill_type=FillType.FULL,
            status=FillStatus.FILLED,
            executed_at=datetime.now(),
            metadata={
                'market_price': float(market_price),
                'order_type': order.order_type.name
            }
        )
        
        return fill
    
    def _can_fill_order(self, order: Order, market_price: Decimal) -> tuple[bool, Optional[Decimal]]:
        """Check if order can be filled and return fill price."""
        if order.order_type == OrderType.MARKET:
            return True, market_price
        
        elif order.order_type == OrderType.LIMIT:
            limit_price = Decimal(str(order.price))
            if order.side == OrderSide.BUY and market_price <= limit_price:
                return True, min(market_price, limit_price)
            elif order.side == OrderSide.SELL and market_price >= limit_price:
                return True, max(market_price, limit_price)
            return False, None
        
        elif order.order_type == OrderType.STOP:
            stop_price = Decimal(str(order.stop_price))
            if order.side == OrderSide.BUY and market_price >= stop_price:
                return True, market_price
            elif order.side == OrderSide.SELL and market_price <= stop_price:
                return True, market_price
            return False, None
        
        elif order.order_type == OrderType.STOP_LIMIT:
            stop_price = Decimal(str(order.stop_price))
            limit_price = Decimal(str(order.price))
            
            # Check if stop triggered
            stop_triggered = False
            if order.side == OrderSide.BUY and market_price >= stop_price:
                stop_triggered = True
            elif order.side == OrderSide.SELL and market_price <= stop_price:
                stop_triggered = True
            
            if not stop_triggered:
                return False, None
            
            # Check limit price
            if order.side == OrderSide.BUY and market_price <= limit_price:
                return True, min(market_price, limit_price)
            elif order.side == OrderSide.SELL and market_price >= limit_price:
                return True, max(market_price, limit_price)
            return False, None
        
        return False, None
    
    def _calculate_slippage(self, order: Order, market_price: Decimal, fill_price: Decimal) -> Decimal:
        """Calculate slippage for the fill."""
        base_slippage = market_price * self._slippage_rate
        
        if order.side == OrderSide.BUY:
            return base_slippage
        else:
            return -base_slippage
    
    def _calculate_commission(self, order: Order, fill_price: Decimal) -> Decimal:
        """Calculate commission for the fill."""
        trade_value = Decimal(str(order.quantity)) * fill_price
        return trade_value * self._commission_rate
    
    async def _update_market_data(self, market_data: Dict[str, Any]) -> None:
        """Update internal market data cache."""
        prices = market_data.get("prices", {})
        for symbol, price in prices.items():
            self._market_prices[symbol] = Decimal(str(price))
    
    async def _update_portfolio_state(self, fill: Fill) -> None:
        """Update portfolio state with fill information."""
        # Convert fill to portfolio update
        quantity_delta = Decimal(str(fill.quantity))
        if fill.side == OrderSide.SELL:
            quantity_delta = -quantity_delta
        
        # Update portfolio state through Risk module
        self._portfolio_state.update_position(
            symbol=fill.symbol,
            quantity_delta=quantity_delta,
            price=Decimal(str(fill.price)),
            timestamp=fill.executed_at
        )
        
        # Update cash for commission
        commission = Decimal(str(fill.commission))
        if commission > 0:
            # Direct access to cash balance (this is a simplification)
            # In practice, this should go through a proper portfolio interface
            self._portfolio_state._cash_balance -= commission
    
    async def _cancel_all_pending_orders(self) -> None:
        """Cancel all pending orders during shutdown."""
        async with self._order_lock:
            order_ids = list(self._pending_orders.keys())
            for order_id in order_ids:
                await self.cancel_order(order_id)


class ValidationResult:
    """Result of order validation."""
    
    def __init__(self, is_valid: bool, reason: str):
        self.is_valid = is_valid
        self.reason = reason


def create_backtest_broker(
    component_id: str,
    portfolio_state: PortfolioStateProtocol,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005
) -> BacktestBrokerRefactored:
    """Factory function to create backtest broker with proper dependencies."""
    return BacktestBrokerRefactored(
        component_id=component_id,
        portfolio_state=portfolio_state,
        commission_rate=Decimal(str(commission_rate)),
        slippage_rate=Decimal(str(slippage_rate))
    )
