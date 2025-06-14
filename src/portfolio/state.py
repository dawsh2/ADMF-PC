"""Portfolio state tracking and management."""

from decimal import Decimal
from datetime import datetime
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import statistics

from .protocols import (
    PortfolioStateProtocol,
    Position,
    RiskMetrics,
)
from ..core.events.types import Event
from ..execution.protocols import OrderProcessor


class PortfolioState(PortfolioStateProtocol):
    """Track and manage portfolio state."""
    
    def __init__(
        self,
        initial_capital: Decimal = Decimal("100000"),
        base_currency: str = "USD",
        order_manager: Optional[OrderProcessor] = None
    ):
        """Initialize portfolio state.
        
        Args:
            initial_capital: Starting capital
            base_currency: Base currency
            order_manager: Order manager for pending order queries (single source of truth)
        """
        self._initial_capital = initial_capital
        self._cash_balance = initial_capital
        self._base_currency = base_currency
        
        # Position tracking
        self._positions: Dict[str, Position] = {}
        
        # Order manager dependency injection - single source of truth for order state
        self._order_manager = order_manager
        
        # P&L tracking
        self._realized_pnl = Decimal(0)
        self._commission_paid = Decimal(0)
        
        # High water mark for drawdown
        self._high_water_mark = initial_capital
        self._max_drawdown = Decimal(0)
        
        # Historical values for risk calculations
        self._value_history: List[Decimal] = [initial_capital]
        self._returns_history: List[Decimal] = []
        
        # Last update timestamp
        self._last_update = datetime.now()
        
        # Track order fill commission
        self._last_commission = Decimal(0)
        
        # Container reference for event publishing
        self._container = None
    
    def set_container(self, container: Any) -> None:
        """Set container reference for event publishing.
        
        Args:
            container: The container that owns this portfolio state
        """
        self._container = container
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self._positions.get(symbol)
    
    def can_create_order(self, symbol: str) -> bool:
        """Check if we should create a new order for this symbol.
        
        This prevents race conditions by ensuring we don't create multiple
        orders for the same symbol while one is pending.
        
        Args:
            symbol: The symbol to check
            
        Returns:
            True if no pending orders exist for this symbol
        """
        if not self._order_manager:
            # Fallback: if no order manager, allow order creation
            return True
            
        # Delegate to OrderManager - single source of truth
        pending_orders = self._order_manager.get_pending_orders()
        pending_for_symbol = [
            order_id for order_id in pending_orders
            if self._order_manager.get_order(order_id) and 
               self._order_manager.get_order(order_id).symbol == symbol
        ]
        return len(pending_for_symbol) == 0
    
    def add_pending_order(self, order_event: Event) -> None:
        """Add order to pending tracking - delegated to OrderManager.
        
        Args:
            order_event: Order event to track as pending
        """
        # Order tracking delegated to OrderManager - single source of truth
        # This method is kept for backward compatibility but delegates responsibility
        if self._order_manager:
            # OrderManager handles this internally when orders are submitted
            pass
    
    def remove_pending_order(self, order_id: str) -> Optional[Event]:
        """Remove order from pending tracking - delegated to OrderManager.
        
        Args:
            order_id: ID of order to remove
            
        Returns:
            None (OrderManager handles state internally)
        """
        # Order tracking delegated to OrderManager - single source of truth
        return None
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[str]:
        """Get pending order IDs, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of pending order IDs
        """
        if not self._order_manager:
            return []
            
        # Delegate to OrderManager - single source of truth
        pending_orders = self._order_manager.get_pending_orders()
        
        if symbol is None:
            return list(pending_orders)
        else:
            # Filter by symbol
            return [
                order_id for order_id in pending_orders
                if self._order_manager.get_order(order_id) and 
                   self._order_manager.get_order(order_id).symbol == symbol
            ]
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self._positions.copy()
    
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        return self._cash_balance
    
    def get_total_value(self) -> Decimal:
        """Get total portfolio value."""
        positions_value = sum(
            pos.market_value for pos in self._positions.values()
        )
        return self._cash_balance + positions_value
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        # Calculate current values
        total_value = self.get_total_value()
        positions_value = sum(
            pos.market_value for pos in self._positions.values()
        )
        unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self._positions.values()
        )
        
        # Update high water mark and drawdown
        if total_value > self._high_water_mark:
            self._high_water_mark = total_value
        
        current_drawdown = Decimal(0)
        if self._high_water_mark > 0:
            current_drawdown = (self._high_water_mark - total_value) / self._high_water_mark
        
        if current_drawdown > self._max_drawdown:
            self._max_drawdown = current_drawdown
        
        # Calculate Sharpe ratio if we have enough history
        sharpe_ratio = None
        if len(self._returns_history) >= 20:  # Need reasonable sample
            try:
                mean_return = statistics.mean(self._returns_history)
                std_return = statistics.stdev(self._returns_history)
                if std_return > 0:
                    # Annualized Sharpe (assuming daily returns)
                    sharpe_ratio = Decimal(str(mean_return / std_return * 252 ** 0.5))
            except:
                pass
        
        # Calculate simple VaR (would need more sophisticated calculation)
        var_95 = None
        if len(self._returns_history) >= 20:
            sorted_returns = sorted(self._returns_history)
            percentile_index = int(len(sorted_returns) * 0.05)
            var_95 = abs(sorted_returns[percentile_index]) * total_value
        
        # Calculate leverage
        leverage = positions_value / total_value if total_value > 0 else Decimal(0)
        
        # Calculate concentration
        concentration = {}
        if total_value > 0:
            for symbol, pos in self._positions.items():
                concentration[symbol] = pos.market_value / total_value
        
        return RiskMetrics(
            total_value=total_value,
            cash_balance=self._cash_balance,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            max_drawdown=self._max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            leverage=leverage,
            concentration=concentration,
            timestamp=datetime.now()
        )
    
    def update_position(
        self,
        symbol: str,
        quantity_delta: Decimal,
        price: Decimal,
        timestamp: datetime
    ) -> Position:
        """Update position with a trade.
        
        Args:
            symbol: Symbol traded
            quantity_delta: Change in quantity (+ for buy, - for sell)
            price: Execution price
            timestamp: Trade timestamp
            
        Returns:
            Updated position
        """
        # Get or create position
        position = self._positions.get(symbol)
        
        if position is None:
            # New position
            position = Position(
                symbol=symbol,
                quantity=quantity_delta,
                average_price=price,
                current_price=price,
                unrealized_pnl=Decimal(0),
                realized_pnl=Decimal(0),
                opened_at=timestamp,
                last_updated=timestamp,
                metadata={}
            )
            self._positions[symbol] = position
        else:
            # Update existing position
            old_quantity = position.quantity
            new_quantity = old_quantity + quantity_delta
            
            if new_quantity == 0:
                # Position closed
                realized = (price - position.average_price) * old_quantity
                self._realized_pnl += realized
                del self._positions[symbol]
                
                # Update cash balance for the closing trade BEFORE returning
                cash_delta = -quantity_delta * price
                self._cash_balance += cash_delta
                
                # Update value history
                self._update_value_history()
                
                # Return closed position for reference
                position.quantity = Decimal(0)
                position.realized_pnl += realized
                position.unrealized_pnl = Decimal(0)
                position.last_updated = timestamp
                return position
            
            elif (old_quantity > 0 and quantity_delta > 0) or \
                 (old_quantity < 0 and quantity_delta < 0):
                # Adding to position
                total_cost = position.average_price * old_quantity + price * quantity_delta
                position.average_price = total_cost / new_quantity
                position.quantity = new_quantity
            
            else:
                # Partial close
                if abs(quantity_delta) > abs(old_quantity):
                    # Flipping position
                    realized = (price - position.average_price) * old_quantity
                    self._realized_pnl += realized
                    position.realized_pnl += realized
                    
                    # New position in opposite direction
                    position.quantity = new_quantity
                    position.average_price = price
                    position.opened_at = timestamp
                else:
                    # Reducing position
                    realized = (price - position.average_price) * (-quantity_delta)
                    self._realized_pnl += realized
                    position.realized_pnl += realized
                    position.quantity = new_quantity
            
            # Update unrealized P&L
            position.unrealized_pnl = (position.current_price - position.average_price) * position.quantity
            position.last_updated = timestamp
        
        # Update cash balance - ensure all values are Decimal
        cash_delta = -quantity_delta * price  # Both should already be Decimal
        self._cash_balance += cash_delta
        
        # Apply commission if set (would be set by fill event handler)
        if self._last_commission > 0:
            self._cash_balance -= self._last_commission
            self._commission_paid += self._last_commission
            self._last_commission = Decimal(0)
        
        # Update value history
        self._update_value_history()
        
        return position
    
    def update_market_prices(self, prices: Dict[str, Decimal]) -> None:
        """Update market prices for positions.
        
        Args:
            prices: Dictionary of symbol -> current price
        """
        for symbol, price in prices.items():
            if symbol in self._positions:
                position = self._positions[symbol]
                position.current_price = price
                position.unrealized_pnl = (price - position.average_price) * position.quantity
                position.last_updated = datetime.now()
        
        # Update value history
        self._update_value_history()
    
    def _update_value_history(self) -> None:
        """Update value history for risk calculations."""
        current_value = self.get_total_value()
        self._value_history.append(current_value)
        
        # Calculate return if we have previous value
        if len(self._value_history) >= 2:
            prev_value = self._value_history[-2]
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value
                self._returns_history.append(daily_return)
        
        # Keep only recent history (e.g., 252 trading days)
        max_history = 252
        if len(self._value_history) > max_history:
            self._value_history = self._value_history[-max_history:]
        if len(self._returns_history) > max_history:
            self._returns_history = self._returns_history[-max_history:]
        
        self._last_update = datetime.now()
    
    def process_event(self, event: Event) -> None:
        """Process an event (fill, market data, etc)."""
        if event.event_type == "FILL":
            # Handle fill event
            payload = event.payload
            symbol = payload["symbol"]
            side = payload["side"]
            quantity = Decimal(str(payload["quantity"]))
            price = Decimal(str(payload["price"]))
            
            # Convert side to quantity delta
            if side.upper() == "BUY":
                quantity_delta = quantity
            else:
                quantity_delta = -quantity
            
            # Update position
            self.update_position(symbol, quantity_delta, price, event.timestamp)
            
            # Remove pending order if this was from an order
            order_id = payload.get("order_id")
            if order_id:
                self.remove_pending_order(order_id)
                
        elif event.event_type == "SIGNAL":
            # Handle signal event - for now just log it
            payload = event.payload
            symbol = payload.get("symbol")
            direction = payload.get("direction")
            strength = payload.get("strength")
            strategy_id = payload.get("strategy_id")
            
            # Log signal reception for debugging
            print(f"📨 Portfolio received SIGNAL: {symbol} {direction} strength={strength} from strategy_id={strategy_id}")
            
            # TODO: Implement signal processing logic
            # - Generate orders based on signals
            # - Apply position sizing
            # - Check risk limits
            
        elif event.event_type == "BAR" or event.event_type == "TICK":
            # Handle market data update
            payload = event.payload
            symbol = payload["symbol"]
            price = None
            
            if "close" in payload:  # Bar data
                price = Decimal(str(payload["close"]))
            elif "price" in payload:  # Tick data  
                price = Decimal(str(payload["price"]))
                
            if price and symbol in self._positions:
                self.update_market_prices({symbol: price})
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        metrics = self.get_risk_metrics()
        initial = self._initial_capital
        current = metrics.total_value
        
        total_return = (current - initial) / initial if initial > 0 else Decimal(0)
        
        return {
            "initial_capital": str(initial),
            "current_value": str(current),
            "total_return": f"{total_return:.2%}",
            "realized_pnl": str(metrics.realized_pnl),
            "unrealized_pnl": str(metrics.unrealized_pnl),
            "commission_paid": str(self._commission_paid),
            "max_drawdown": f"{metrics.max_drawdown:.2%}",
            "current_drawdown": f"{metrics.current_drawdown:.2%}",
            "sharpe_ratio": str(metrics.sharpe_ratio) if metrics.sharpe_ratio else "N/A",
            "positions_count": len(self._positions),
            "pending_orders_count": len(self._pending_orders),
            "last_update": self._last_update.isoformat()
        }
