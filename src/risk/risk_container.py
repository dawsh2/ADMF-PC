"""
File: src/risk/risk_container.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#risk-container
Step: 2 - Add Risk Container
Dependencies: core.events, core.logging, risk.models

Risk Container implementation for Step 2.
Encapsulates risk management logic with event isolation.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from ..core.logging.structured import ContainerLogger
from ..core.events.enhanced_isolation import get_enhanced_isolation_manager
from .step2_portfolio_state import PortfolioState
from .step2_position_sizer import PositionSizer
from .step2_risk_limits import RiskLimits
from .step2_order_manager import OrderManager
from .models import RiskConfig, TradingSignal, Order, Fill


class RiskContainer:
    """
    Encapsulates all risk management components.
    
    Maintains portfolio state and enforces risk limits in an isolated container.
    Transforms trading signals into risk-adjusted orders.
    
    Architecture Context:
        - Part of: Step 2 - Add Risk Container
        - Implements: Container isolation with event-driven risk management
        - Provides: Position sizing, risk limits, portfolio tracking
        - Dependencies: Enhanced event isolation, structured logging
    
    Example:
        config = RiskConfig(
            sizing_method='fixed',
            initial_capital=100000,
            max_position_size=0.1
        )
        container = RiskContainer("risk_001", config)
        container.on_signal(trading_signal)
    """
    
    def __init__(self, container_id: str, config: RiskConfig):
        """
        Initialize risk container with isolated components.
        
        Args:
            container_id: Unique container identifier
            config: Risk configuration parameters
        """
        self.container_id = container_id
        self.config = config
        
        # Create isolated event bus
        self.isolation_manager = get_enhanced_isolation_manager()
        self.event_bus = self.isolation_manager.create_container_bus(
            f"{container_id}_risk"
        )
        
        # Initialize components
        self.portfolio_state = PortfolioState(container_id, config.initial_capital)
        self.position_sizer = PositionSizer(config.sizing_method, config)
        self.risk_limits = RiskLimits(config)
        self.order_manager = OrderManager(container_id)
        
        # Setup logging
        self.logger = ContainerLogger("RiskContainer", container_id, "risk_container")
        
        # State tracking
        self.processed_signals = 0
        self.created_orders = 0
        self.rejected_signals = 0
        
        # Wire internal events
        self._setup_internal_events()
        
        self.logger.info(
            "RiskContainer initialized",
            container_id=container_id,
            initial_capital=config.initial_capital,
            sizing_method=config.sizing_method
        )
    
    def on_signal(self, signal: TradingSignal) -> None:
        """
        Process trading signal through risk pipeline.
        
        This method implements the core risk management workflow:
        1. Log signal receipt
        2. Check risk limits
        3. Calculate position size
        4. Create order if valid
        5. Update portfolio state
        6. Emit order event
        
        Args:
            signal: Trading signal to process
        """
        self.processed_signals += 1
        
        self.logger.log_event_flow(
            "SIGNAL_RECEIVED", "strategy", "risk", 
            f"Signal: {signal.side.value} {signal.symbol} strength={signal.strength}"
        )
        
        # Check risk limits
        if not self.risk_limits.can_trade(self.portfolio_state, signal):
            self.rejected_signals += 1
            self.logger.warning(
                "Signal rejected by risk limits",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                side=signal.side.value,
                reason="Risk limits exceeded"
            )
            return
        
        # Calculate position size
        size = self.position_sizer.calculate_size(signal, self.portfolio_state)
        
        if size <= 0:
            self.rejected_signals += 1
            self.logger.warning(
                "Signal rejected - zero size",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                calculated_size=size
            )
            return
        
        # Create order
        order = self.order_manager.create_order(
            signal, size, self.portfolio_state.get_current_prices()
        )
        
        if order:
            self.created_orders += 1
            
            # Update portfolio state optimistically
            self.portfolio_state.add_pending_order(order)
            
            self.logger.log_event_flow(
                "ORDER_CREATED", "risk", "execution",
                f"Order: {order.side.value} {order.quantity} {order.symbol}"
            )
            
            # Emit order event
            self.event_bus.publish("ORDER", order)
            
            self.logger.info(
                "Order created and published",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value,
                quantity=float(order.quantity)
            )
    
    def on_fill(self, fill: Fill) -> None:
        """
        Process fill event and update portfolio state.
        
        Args:
            fill: Fill event from execution engine
        """
        self.logger.log_event_flow(
            "FILL_RECEIVED", "execution", "risk",
            f"Fill: {fill.side.value} {fill.quantity} {fill.symbol} @ {fill.price}"
        )
        
        # Update portfolio state
        self.portfolio_state.update_position(fill)
        
        # Remove from pending orders
        self.portfolio_state.remove_pending_order(fill.order_id)
        
        self.logger.info(
            "Portfolio updated from fill",
            fill_id=fill.fill_id,
            symbol=fill.symbol,
            quantity=float(fill.quantity),
            price=float(fill.price),
            new_cash=float(self.portfolio_state.cash),
            total_value=float(self.portfolio_state.total_value)
        )
    
    def update_market_data(self, market_data: Dict[str, float]) -> None:
        """
        Update portfolio with current market prices.
        
        Args:
            market_data: Dictionary of symbol -> current price
        """
        self.portfolio_state.update_prices(market_data)
        total_value = self.portfolio_state.calculate_total_value()
        
        self.logger.trace(
            "Portfolio value updated",
            total_value=float(total_value),
            cash=float(self.portfolio_state.cash),
            positions_count=len(self.portfolio_state.positions)
        )
    
    def _setup_internal_events(self) -> None:
        """Setup internal event subscriptions."""
        # Subscribe to fill events to update portfolio
        self.event_bus.subscribe("FILL", self.on_fill)
        
        # Subscribe to market data updates
        self.event_bus.subscribe("MARKET_DATA", self.update_market_data)
        
        self.logger.debug("Internal event subscriptions configured")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current container state for debugging and monitoring.
        
        Returns:
            Dictionary containing container state information
        """
        return {
            "container_id": self.container_id,
            "processed_signals": self.processed_signals,
            "created_orders": self.created_orders,
            "rejected_signals": self.rejected_signals,
            "portfolio_value": float(self.portfolio_state.total_value),
            "cash": float(self.portfolio_state.cash),
            "positions_count": len(self.portfolio_state.positions),
            "pending_orders": len(self.portfolio_state.pending_orders),
            "risk_config": {
                "sizing_method": self.config.sizing_method,
                "initial_capital": self.config.initial_capital,
                "max_position_size": self.config.max_position_size,
                "max_portfolio_risk": self.config.max_portfolio_risk
            }
        }
    
    def reset(self) -> None:
        """
        Reset container state for new calculation cycle.
        
        This method supports backtesting scenarios where risk containers
        need to be reset between test runs.
        """
        self.portfolio_state.reset(self.config.initial_capital)
        self.processed_signals = 0
        self.created_orders = 0
        self.rejected_signals = 0
        
        self.logger.info("RiskContainer reset")
    
    def cleanup(self) -> None:
        """
        Cleanup container resources.
        
        This method should be called when the container is no longer needed
        to properly release event bus resources and log final state.
        """
        self.logger.info(
            "RiskContainer cleanup",
            final_state=self.get_state()
        )
        
        # Remove from isolation manager
        if self.isolation_manager:
            self.isolation_manager.remove_container_bus(f"{self.container_id}_risk")