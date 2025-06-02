"""
File: src/risk/step2_portfolio_state.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#portfolio-state
Step: 2 - Add Risk Container
Dependencies: core.logging, risk.models

Portfolio state tracking for Step 2 risk container.
Maintains positions, cash, and portfolio metrics.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal

from ..core.logging.structured import ContainerLogger
from .models import Fill, Order, OrderSide


class Position:
    """
    Represents a position in a single symbol.
    
    Architecture Context:
        - Part of: Step 2 - Add Risk Container
        - Implements: Protocol-based position tracking without inheritance
        - Provides: Position state and P&L calculation
        - Dependencies: None (simple data structure)
    """
    
    def __init__(self, symbol: str, quantity: Decimal = Decimal('0'), avg_price: Decimal = Decimal('0')):
        """
        Initialize position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity (positive = long, negative = short)
            avg_price: Average entry price
        """
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.realized_pnl = Decimal('0')
        self.last_updated = datetime.now()
    
    def update_from_fill(self, fill: Fill) -> None:
        """
        Update position from a fill.
        
        Args:
            fill: Fill to apply to position
        """
        fill_quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        
        if self.quantity == 0:
            # Opening position
            self.quantity = fill_quantity
            self.avg_price = fill.price
        elif (self.quantity > 0 and fill_quantity > 0) or (self.quantity < 0 and fill_quantity < 0):
            # Adding to position
            total_value = (self.quantity * self.avg_price) + (fill_quantity * fill.price)
            self.quantity += fill_quantity
            self.avg_price = total_value / self.quantity if self.quantity != 0 else Decimal('0')
        else:
            # Reducing or reversing position
            if abs(fill_quantity) >= abs(self.quantity):
                # Closing or reversing position
                closed_quantity = self.quantity
                pnl_per_share = fill.price - self.avg_price
                self.realized_pnl += closed_quantity * pnl_per_share
                
                remaining_quantity = fill_quantity + self.quantity
                if remaining_quantity != 0:
                    # Reversing position
                    self.quantity = remaining_quantity
                    self.avg_price = fill.price
                else:
                    # Closing position
                    self.quantity = Decimal('0')
                    self.avg_price = Decimal('0')
            else:
                # Partial close
                close_ratio = abs(fill_quantity) / abs(self.quantity)
                pnl_per_share = fill.price - self.avg_price
                self.realized_pnl += abs(fill_quantity) * pnl_per_share
                self.quantity += fill_quantity
        
        self.last_updated = datetime.now()
    
    def get_market_value(self, current_price: float) -> Decimal:
        """
        Get current market value of position.
        
        Args:
            current_price: Current market price
            
        Returns:
            Market value of position
        """
        return self.quantity * Decimal(str(current_price))
    
    def get_unrealized_pnl(self, current_price: float) -> Decimal:
        """
        Get unrealized P&L.
        
        Args:
            current_price: Current market price
            
        Returns:
            Unrealized P&L
        """
        if self.quantity == 0:
            return Decimal('0')
        
        return self.quantity * (Decimal(str(current_price)) - self.avg_price)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'avg_price': float(self.avg_price),
            'realized_pnl': float(self.realized_pnl),
            'last_updated': self.last_updated.isoformat()
        }


class PortfolioState:
    """
    Tracks portfolio state for Step 2 risk container.
    
    Maintains cash, positions, and provides portfolio metrics
    for risk management decisions.
    
    Architecture Context:
        - Part of: Step 2 - Add Risk Container
        - Implements: Protocol-based portfolio tracking without inheritance
        - Provides: Real-time portfolio state for risk calculations
        - Dependencies: Structured logging for audit trail
    
    Example:
        portfolio = PortfolioState("risk_001", 100000.0)
        portfolio.update_position(fill)
        total_value = portfolio.calculate_total_value()
    """
    
    def __init__(self, container_id: str, initial_capital: float):
        """
        Initialize portfolio state.
        
        Args:
            container_id: Container ID for logging context
            initial_capital: Starting cash amount
        """
        self.container_id = container_id
        self.initial_capital = Decimal(str(initial_capital))
        self.cash = Decimal(str(initial_capital))
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Order] = {}
        
        # Market data
        self.current_prices: Dict[str, float] = {}
        
        # Portfolio metrics
        self.total_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.current_drawdown = Decimal('0')
        
        # Logging
        self.logger = ContainerLogger("PortfolioState", container_id, "portfolio_state")
        
        self.logger.info(
            "PortfolioState initialized",
            container_id=container_id,
            initial_capital=float(initial_capital)
        )
    
    def update_position(self, fill: Fill) -> None:
        """
        Update position from fill.
        
        Args:
            fill: Fill to apply
        """
        self.logger.trace(
            "Updating position from fill",
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=float(fill.quantity),
            price=float(fill.price)
        )
        
        # Get or create position
        if fill.symbol not in self.positions:
            self.positions[fill.symbol] = Position(fill.symbol)
        
        position = self.positions[fill.symbol]
        old_quantity = position.quantity
        
        # Update position
        position.update_from_fill(fill)
        
        # Update cash
        cash_change = -fill.quantity * fill.price if fill.side == OrderSide.BUY else fill.quantity * fill.price
        self.cash += cash_change
        
        # Clean up zero positions
        if position.quantity == 0:
            del self.positions[fill.symbol]
        
        self.logger.debug(
            "Position updated",
            symbol=fill.symbol,
            old_quantity=float(old_quantity),
            new_quantity=float(position.quantity) if fill.symbol in self.positions else 0,
            cash_change=float(cash_change),
            new_cash=float(self.cash)
        )
        
        # Recalculate portfolio metrics
        self._update_portfolio_metrics()
    
    def add_pending_order(self, order: Order) -> None:
        """
        Add pending order to tracking.
        
        Args:
            order: Order to track
        """
        self.pending_orders[order.order_id] = order
        
        self.logger.trace(
            "Pending order added",
            order_id=order.order_id,
            symbol=order.symbol,
            pending_count=len(self.pending_orders)
        )
    
    def remove_pending_order(self, order_id: str) -> Optional[Order]:
        """
        Remove pending order.
        
        Args:
            order_id: Order ID to remove
            
        Returns:
            Removed order if found
        """
        order = self.pending_orders.pop(order_id, None)
        
        if order:
            self.logger.trace(
                "Pending order removed",
                order_id=order_id,
                pending_count=len(self.pending_orders)
            )
        
        return order
    
    def update_prices(self, market_data: Dict[str, float]) -> None:
        """
        Update current market prices.
        
        Args:
            market_data: Dictionary of symbol -> price
        """
        self.current_prices.update(market_data)
        self._update_portfolio_metrics()
        
        self.logger.trace(
            "Market prices updated",
            symbols_updated=list(market_data.keys()),
            total_symbols=len(self.current_prices)
        )
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current market prices."""
        return self.current_prices.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position if exists, None otherwise
        """
        return self.positions.get(symbol)
    
    def calculate_total_value(self) -> Decimal:
        """
        Calculate total portfolio value.
        
        Returns:
            Total portfolio value including cash and positions
        """
        total = self.cash
        
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol)
            if current_price:
                total += position.get_market_value(current_price)
        
        return total
    
    def get_exposure(self) -> Dict[str, Any]:
        """
        Get portfolio exposure metrics.
        
        Returns:
            Dictionary containing exposure information
        """
        long_exposure = Decimal('0')
        short_exposure = Decimal('0')
        
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol)
            if current_price:
                market_value = position.get_market_value(current_price)
                if position.quantity > 0:
                    long_exposure += market_value
                else:
                    short_exposure += abs(market_value)
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        return {
            'long_exposure': float(long_exposure),
            'short_exposure': float(short_exposure),
            'gross_exposure': float(gross_exposure),
            'net_exposure': float(net_exposure),
            'cash': float(self.cash),
            'total_value': float(self.total_value)
        }
    
    def get_unrealized_pnl(self) -> Decimal:
        """
        Calculate total unrealized P&L.
        
        Returns:
            Total unrealized P&L across all positions
        """
        total_pnl = Decimal('0')
        
        for symbol, position in self.positions.items():
            current_price = self.current_prices.get(symbol)
            if current_price:
                total_pnl += position.get_unrealized_pnl(current_price)
        
        return total_pnl
    
    def get_realized_pnl(self) -> Decimal:
        """
        Calculate total realized P&L.
        
        Returns:
            Total realized P&L across all positions
        """
        total_pnl = Decimal('0')
        
        for position in self.positions.values():
            total_pnl += position.realized_pnl
        
        return total_pnl
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio metrics like drawdown."""
        old_total = self.total_value
        self.total_value = self.calculate_total_value()
        
        # Update peak value
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
            self.current_drawdown = Decimal('0')
        else:
            # Calculate drawdown as percentage
            if self.peak_value > 0:
                self.current_drawdown = (self.peak_value - self.total_value) / self.peak_value
        
        if self.total_value != old_total:
            self.logger.trace(
                "Portfolio metrics updated",
                old_total=float(old_total),
                new_total=float(self.total_value),
                peak_value=float(self.peak_value),
                current_drawdown=float(self.current_drawdown)
            )
    
    def reset(self, new_initial_capital: float) -> None:
        """
        Reset portfolio state.
        
        Args:
            new_initial_capital: New starting capital
        """
        positions_cleared = len(self.positions)
        orders_cleared = len(self.pending_orders)
        
        self.initial_capital = Decimal(str(new_initial_capital))
        self.cash = Decimal(str(new_initial_capital))
        self.positions.clear()
        self.pending_orders.clear()
        self.current_prices.clear()
        
        self.total_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.current_drawdown = Decimal('0')
        
        self.logger.info(
            "Portfolio state reset",
            new_capital=new_initial_capital,
            positions_cleared=positions_cleared,
            orders_cleared=orders_cleared
        )
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Dictionary containing portfolio state
        """
        positions_data = {}
        for symbol, position in self.positions.items():
            positions_data[symbol] = position.to_dict()
            current_price = self.current_prices.get(symbol)
            if current_price:
                positions_data[symbol]['current_price'] = current_price
                positions_data[symbol]['market_value'] = float(position.get_market_value(current_price))
                positions_data[symbol]['unrealized_pnl'] = float(position.get_unrealized_pnl(current_price))
        
        return {
            'container_id': self.container_id,
            'cash': float(self.cash),
            'total_value': float(self.total_value),
            'peak_value': float(self.peak_value),
            'current_drawdown': float(self.current_drawdown),
            'positions': positions_data,
            'pending_orders_count': len(self.pending_orders),
            'exposure': self.get_exposure(),
            'unrealized_pnl': float(self.get_unrealized_pnl()),
            'realized_pnl': float(self.get_realized_pnl()),
            'total_return': float((self.total_value - self.initial_capital) / self.initial_capital) if self.initial_capital > 0 else 0.0
        }