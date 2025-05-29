"""Unified Risk & Portfolio management container."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional, Set
from threading import RLock

from ..core.containers.universal import UniversalScopedContainer
from ..core.events.types import EventType, Event
from ..core.coordinator.types import ExecutionContext
from ..core.logging.structured import get_logger
from ..core.components.protocols import Capability
from ..strategy.protocols import StrategyProtocol

from .protocols import (
    RiskPortfolioProtocol,
    PositionSizerProtocol,
    RiskLimitProtocol,
    PortfolioStateProtocol,
    SignalProcessorProtocol,
    Signal,
    Order,
    RiskMetrics,
    RiskCapability,
    PortfolioTrackingCapability,
    PositionSizingCapability,
    RiskLimitCapability,
)
from .portfolio_state import PortfolioState
from .signal_processing import SignalProcessor
from .position_sizing import PercentagePositionSizer


class RiskPortfolioContainer(UniversalScopedContainer, RiskPortfolioProtocol):
    """Unified Risk & Portfolio management container.
    
    This container manages:
    - Multiple strategy components
    - Signal to order conversion
    - Position sizing
    - Risk limit enforcement
    - Portfolio state tracking
    - Thread-safe operations based on execution context
    """
    
    def __init__(
        self,
        name: str = "RiskPortfolio",
        initial_capital: Decimal = Decimal("100000"),
        base_currency: str = "USD"
    ):
        """Initialize Risk & Portfolio container.
        
        Args:
            name: Container name
            initial_capital: Starting capital
            base_currency: Base currency for portfolio
        """
        super().__init__(container_id=name, container_type="risk_portfolio")
        self.name = name  # Store name separately
        self.logger = get_logger(self.__class__.__name__)
        
        # Core components
        self._portfolio_state = PortfolioState(
            initial_capital=initial_capital,
            base_currency=base_currency
        )
        self._signal_processor = SignalProcessor()
        self._position_sizer: PositionSizerProtocol = PercentagePositionSizer(
            percentage=Decimal("0.02")  # Default 2% per position
        )
        
        # Risk limits and strategies
        self._risk_limits: List[RiskLimitProtocol] = []
        self._strategies: Dict[str, StrategyProtocol] = {}
        
        # Thread safety (used when execution context requires it)
        self._lock = RLock()
        
        # Order tracking
        self._order_history: List[Order] = []
        self._active_orders: Dict[str, Order] = {}
        
        # Register capabilities
        self._register_capabilities()
    
    def _register_capabilities(self) -> None:
        """Register risk management capabilities."""
        self.add_capability(RiskCapability())
        self.add_capability(PortfolioTrackingCapability())
        self.add_capability(PositionSizingCapability())
        self.add_capability(RiskLimitCapability())
    
    def get_required_capabilities(self) -> Set[type[Capability]]:
        """Get required capabilities."""
        return {RiskCapability}
    
    def get_provided_capabilities(self) -> Set[type[Capability]]:
        """Get provided capabilities."""
        return {
            RiskCapability,
            PortfolioTrackingCapability,
            PositionSizingCapability,
            RiskLimitCapability,
        }
    
    async def start(self) -> None:
        """Start the container."""
        await super().start()
        self.logger.info(
            "risk_portfolio_started",
            initial_capital=str(self._portfolio_state.get_cash_balance()),
            base_currency=self._portfolio_state._base_currency
        )
    
    async def stop(self) -> None:
        """Stop the container."""
        # Generate final risk report
        final_report = self.get_risk_report()
        self.logger.info("risk_portfolio_stopping", risk_report=final_report)
        
        await super().stop()
    
    def add_strategy(self, strategy: StrategyProtocol) -> None:
        """Add a strategy component.
        
        Args:
            strategy: Strategy to manage
        """
        strategy_id = strategy.get_metadata().get("id", str(uuid.uuid4()))
        self._strategies[strategy_id] = strategy
        
        self.logger.info(
            "strategy_added",
            strategy_id=strategy_id,
            strategy_type=type(strategy).__name__
        )
    
    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy component.
        
        Args:
            strategy_id: ID of strategy to remove
        """
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            self.logger.info("strategy_removed", strategy_id=strategy_id)
    
    def process_signals(
        self,
        signals: List[Signal],
        market_data: Dict[str, Any]
    ) -> List[Order]:
        """Process multiple signals into orders.
        
        Thread-safe based on execution context.
        
        Args:
            signals: List of trading signals
            market_data: Current market data
            
        Returns:
            List of approved orders
        """
        context = self.get_execution_context()
        
        if context and context.is_concurrent:
            with self._lock:
                return self._process_signals_impl(signals, market_data)
        else:
            return self._process_signals_impl(signals, market_data)
    
    def _process_signals_impl(
        self,
        signals: List[Signal],
        market_data: Dict[str, Any]
    ) -> List[Order]:
        """Implementation of signal processing."""
        orders = []
        
        # Sort signals by priority (exits before entries)
        sorted_signals = sorted(
            signals,
            key=lambda s: 0 if s.signal_type.value in ["exit", "risk_exit"] else 1
        )
        
        for signal in sorted_signals:
            try:
                # Process signal
                order = self._signal_processor.process_signal(
                    signal=signal,
                    portfolio_state=self._portfolio_state,
                    position_sizer=self._position_sizer,
                    risk_limits=self._risk_limits,
                    market_data=market_data
                )
                
                if order:
                    orders.append(order)
                    self._active_orders[order.order_id] = order
                    
                    # Emit order created event
                    self._emit_event(
                        EventType.ORDER,
                        Event(
                            event_type=EventType.ORDER,
                            source_id=self.name,
                            payload={
                                "type": "order_created",
                                "order": order,
                                "signal": signal
                            }
                        )
                    )
                    
                    self.logger.info(
                        "order_created",
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=str(order.quantity),
                        risk_checks=order.risk_checks_passed
                    )
                else:
                    self.logger.warning(
                        "signal_rejected",
                        signal=signal,
                        reason="Failed risk checks or sizing"
                    )
                    
            except Exception as e:
                self.logger.error(
                    "signal_processing_error",
                    signal=signal,
                    error=str(e)
                )
        
        return orders
    
    def get_portfolio_state(self) -> PortfolioStateProtocol:
        """Get current portfolio state."""
        return self._portfolio_state
    
    def update_fills(self, fills: List[Dict[str, Any]]) -> None:
        """Update portfolio with executed fills.
        
        Expected fill format:
        {
            "order_id": str,
            "symbol": str,
            "side": str,
            "quantity": Decimal,
            "price": Decimal,
            "timestamp": datetime,
            "commission": Decimal
        }
        """
        context = self.get_execution_context()
        
        if context and context.is_concurrent:
            with self._lock:
                self._update_fills_impl(fills)
        else:
            self._update_fills_impl(fills)
    
    def _update_fills_impl(self, fills: List[Dict[str, Any]]) -> None:
        """Implementation of fill updates."""
        for fill in fills:
            try:
                # Update portfolio state
                quantity_delta = fill["quantity"]
                if fill["side"] == "sell":
                    quantity_delta = -quantity_delta
                
                position = self._portfolio_state.update_position(
                    symbol=fill["symbol"],
                    quantity_delta=quantity_delta,
                    price=fill["price"],
                    timestamp=fill["timestamp"]
                )
                
                # Update cash for commission
                commission = fill.get("commission", Decimal(0))
                if commission:
                    self._portfolio_state._cash_balance -= commission
                
                # Move order to history
                order_id = fill.get("order_id")
                if order_id and order_id in self._active_orders:
                    order = self._active_orders.pop(order_id)
                    self._order_history.append(order)
                
                # Emit fill event
                self._emit_event(
                    EventType.FILL,
                    Event(
                        event_type=EventType.FILL,
                        source_id=self.name,
                        payload={
                            "type": "fill_processed",
                            "fill": fill,
                            "position": position
                        }
                    )
                )
                
                self.logger.info(
                    "fill_processed",
                    symbol=fill["symbol"],
                    side=fill["side"],
                    quantity=str(fill["quantity"]),
                    price=str(fill["price"])
                )
                
            except Exception as e:
                self.logger.error(
                    "fill_processing_error",
                    fill=fill,
                    error=str(e)
                )
    
    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """Update market data for risk calculations.
        
        Expected format:
        {
            "prices": {symbol: price},
            "timestamp": datetime,
            ...
        }
        """
        prices = market_data.get("prices", {})
        if prices:
            decimal_prices = {
                symbol: Decimal(str(price))
                for symbol, price in prices.items()
            }
            self._portfolio_state.update_market_prices(decimal_prices)
    
    def add_risk_limit(self, limit: RiskLimitProtocol) -> None:
        """Add a risk limit."""
        self._risk_limits.append(limit)
        self.logger.info(
            "risk_limit_added",
            limit_type=type(limit).__name__,
            limit_info=limit.get_limit_info()
        )
    
    def remove_risk_limit(self, limit_type: type) -> None:
        """Remove a risk limit by type."""
        self._risk_limits = [
            limit for limit in self._risk_limits
            if not isinstance(limit, limit_type)
        ]
        self.logger.info(
            "risk_limit_removed",
            limit_type=limit_type.__name__
        )
    
    def set_position_sizer(self, sizer: PositionSizerProtocol) -> None:
        """Set position sizing strategy."""
        self._position_sizer = sizer
        self.logger.info(
            "position_sizer_set",
            sizer_type=type(sizer).__name__
        )
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        metrics = self._portfolio_state.get_risk_metrics()
        
        return {
            "portfolio_metrics": {
                "total_value": str(metrics.total_value),
                "cash_balance": str(metrics.cash_balance),
                "positions_value": str(metrics.positions_value),
                "unrealized_pnl": str(metrics.unrealized_pnl),
                "realized_pnl": str(metrics.realized_pnl),
                "max_drawdown": str(metrics.max_drawdown),
                "current_drawdown": str(metrics.current_drawdown),
                "leverage": str(metrics.leverage),
                "timestamp": metrics.timestamp.isoformat()
            },
            "positions": {
                symbol: {
                    "quantity": str(pos.quantity),
                    "market_value": str(pos.market_value),
                    "unrealized_pnl": str(pos.unrealized_pnl),
                    "pnl_percentage": str(pos.pnl_percentage)
                }
                for symbol, pos in self._portfolio_state.get_all_positions().items()
            },
            "risk_limits": [
                {
                    "type": type(limit).__name__,
                    "info": limit.get_limit_info()
                }
                for limit in self._risk_limits
            ],
            "position_sizer": type(self._position_sizer).__name__,
            "active_strategies": list(self._strategies.keys()),
            "active_orders": len(self._active_orders),
            "order_history_count": len(self._order_history)
        }
    
    def _emit_event(self, event_type: EventType, data: Event) -> None:
        """Emit event through container event system."""
        # Use parent container's event system if available
        if self._parent_container and hasattr(self._parent_container, 'publish_event'):
            self._parent_container.publish_event(event_type, data)
        else:
            # Log event if no event system available
            self.logger.debug(
                "event_emitted",
                event_type=event_type.value,
                data=data
            )