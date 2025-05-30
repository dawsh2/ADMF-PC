"""Improved Risk & Portfolio management container with proper dependency injection."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional, Set
from threading import RLock

from ..core.containers.universal import UniversalScopedContainer
from ..core.dependencies.container import DependencyContainer
from ..core.events.types import EventType, Event
from ..core.components.protocols import Component, Lifecycle, EventCapable
import logging

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
from .dependency_injection import (
    RiskDependencyResolver,
    RiskComponentFactory,
    create_position_sizer_spec,
    create_risk_limit_spec,
    create_signal_processor_spec,
)
from .portfolio_state import PortfolioState


class RiskPortfolioContainer(Component, Lifecycle, EventCapable):
    """
    Improved Risk & Portfolio management container with proper dependency injection.
    
    This container manages:
    - Multiple strategy components
    - Signal to order conversion
    - Position sizing
    - Risk limit enforcement
    - Portfolio state tracking
    - Proper dependency injection
    """
    
    def __init__(
        self,
        component_id: str,
        dependency_container: DependencyContainer,
        initial_capital: Decimal = Decimal("100000"),
        base_currency: str = "USD"
    ):
        """Initialize Risk & Portfolio container with proper DI.
        
        Args:
            component_id: Unique component identifier
            dependency_container: DI container for component management
            initial_capital: Starting capital
            base_currency: Base currency for portfolio
        """
        self._component_id = component_id
        self._dependency_container = dependency_container
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create dependency resolver and factory
        self._factory = RiskComponentFactory(dependency_container)
        self._resolver = RiskDependencyResolver(dependency_container, self._factory)
        
        # Core state - injected dependencies will be set during initialization
        self._portfolio_state: Optional[PortfolioStateProtocol] = None
        self._signal_processor: Optional[SignalProcessorProtocol] = None
        self._event_bus = None
        
        # Collections managed by resolver
        self._strategies: Dict[str, Any] = {}
        
        # Thread safety (used when execution context requires it)
        self._lock = RLock()
        
        # Order tracking
        self._order_history: List[Order] = []
        self._active_orders: Dict[str, Order] = {}
        
        # Lifecycle state
        self._initialized = False
        self._running = False
        
        # Setup core portfolio state
        self._setup_portfolio_state(initial_capital, base_currency)
    
    @property
    def component_id(self) -> str:
        """Component identifier."""
        return self._component_id
    
    @property
    def event_bus(self):
        """Get event bus."""
        return self._event_bus
    
    @event_bus.setter
    def event_bus(self, value):
        """Set event bus."""
        self._event_bus = value
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the container with context."""
        self.event_bus = context.get('event_bus')
        
        # Setup default signal processor if not configured
        if not self._resolver._signal_processor:
            processor_spec = create_signal_processor_spec(
                processor_type='standard',
                name='default_processor'
            )
            self._resolver.register_signal_processor(processor_spec, context)
        
        self._initialized = True
        portfolio_state = self._resolver.get_portfolio_state()
        self.logger.info(
            f"Risk portfolio initialized - ID: {self._component_id}, "
            f"Capital: {portfolio_state.get_cash_balance()}"
        )
    
    def start(self) -> None:
        """Start the container."""
        if not self._initialized:
            raise RuntimeError("Container not initialized")
        
        self._running = True
        self.logger.info(f"Risk portfolio started - ID: {self._component_id}")
    
    def stop(self) -> None:
        """Stop the container."""
        self._running = False
        
        # Generate final risk report
        final_report = self.get_risk_report()
        self.logger.info(
            f"Risk portfolio stopped - ID: {self._component_id}, Final report: {final_report}"
        )
    
    def reset(self) -> None:
        """Reset the container state."""
        # Reset portfolio state
        try:
            portfolio_state = self._resolver.get_portfolio_state()
            portfolio_state._cash_balance = portfolio_state._initial_capital
            portfolio_state._positions.clear()
            portfolio_state._realized_pnl = Decimal(0)
            portfolio_state._value_history = [portfolio_state._initial_capital]
            portfolio_state._returns_history = []
        except ValueError:
            # Portfolio state not registered yet
            pass
        
        # Clear orders
        self._order_history.clear()
        self._active_orders.clear()
        
        self.logger.info(f"Risk portfolio reset - ID: {self._component_id}")
    
    def teardown(self) -> None:
        """Teardown the container."""
        # Stop if running
        if self._running:
            self.stop()
        
        # Clear all state
        self._strategies.clear()
        self._order_history.clear()
        self._active_orders.clear()
        
        self.logger.info(f"Risk portfolio torn down - ID: {self._component_id}")
    
    def initialize_events(self) -> None:
        """Initialize event subscriptions."""
        # Event subscriptions would be managed by parent container
        pass
    
    def teardown_events(self) -> None:
        """Clean up event subscriptions."""
        # Event cleanup would be managed by parent container
        pass
    
    def configure_position_sizer(
        self,
        name: str,
        sizer_type: str,
        **params
    ) -> PositionSizerProtocol:
        """Configure a position sizer.
        
        Args:
            name: Sizer name
            sizer_type: Type of sizer (fixed, percentage, etc.)
            **params: Sizer parameters
            
        Returns:
            Created position sizer
        """
        spec = create_position_sizer_spec(sizer_type, name, **params)
        return self._resolver.register_position_sizer(name, spec)
    
    def configure_risk_limit(
        self,
        limit_type: str,
        name: Optional[str] = None,
        **params
    ) -> RiskLimitProtocol:
        """Configure a risk limit.
        
        Args:
            limit_type: Type of limit (position, exposure, etc.)
            name: Optional name for the limit
            **params: Limit parameters
            
        Returns:
            Created risk limit
        """
        if not name:
            name = f"{limit_type}_limit_{len(self._resolver._risk_limits)}"
        
        spec = create_risk_limit_spec(limit_type, name, **params)
        return self._resolver.register_risk_limit(spec)
    
    def configure_signal_processor(
        self,
        processor_type: str = 'standard',
        **params
    ) -> SignalProcessorProtocol:
        """Configure signal processor.
        
        Args:
            processor_type: Type of processor
            **params: Processor parameters
            
        Returns:
            Created signal processor
        """
        spec = create_signal_processor_spec(processor_type, 'signal_processor', **params)
        return self._resolver.register_signal_processor(spec)
    
    def add_strategy(self, strategy: Any) -> None:
        """Add a strategy component.
        
        Args:
            strategy: Strategy to manage
        """
        strategy_id = getattr(strategy, 'component_id', str(uuid.uuid4()))
        self._strategies[strategy_id] = strategy
        
        self.logger.info(
            f"Strategy added - ID: {strategy_id}, Type: {type(strategy).__name__}"
        )
    
    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy component.
        
        Args:
            strategy_id: ID of strategy to remove
        """
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            self.logger.info(f"Strategy removed - ID: {strategy_id}")
    
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
        # Check if threading is needed (simplified for now)
        use_threading = len(signals) > 10  # Heuristic
        
        if use_threading:
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
        
        # Get dependencies
        signal_processor = self._resolver.get_signal_processor()
        portfolio_state = self._resolver.get_portfolio_state()
        risk_limits = self._resolver.get_risk_limits()
        
        # Get default position sizer (create if needed)
        try:
            position_sizer = self._resolver.get_position_sizer('default')
        except ValueError:
            # Create default position sizer
            position_sizer = self.configure_position_sizer(
                'default',
                'percentage',
                percentage=Decimal("2.0")
            )
        
        # Sort signals by priority (exits before entries)
        sorted_signals = sorted(
            signals,
            key=lambda s: 0 if s.signal_type.value in ["exit", "risk_exit"] else 1
        )
        
        for signal in sorted_signals:
            try:
                # Process signal
                order = signal_processor.process_signal(
                    signal=signal,
                    portfolio_state=portfolio_state,
                    position_sizer=position_sizer,
                    risk_limits=risk_limits,
                    market_data=market_data
                )
                
                if order:
                    orders.append(order)
                    self._active_orders[order.order_id] = order
                    
                    # Emit order created event
                    self._emit_event(
                        EventType.ORDER,
                        {
                            "type": "order_created",
                            "order": order,
                            "signal": signal
                        }
                    )
                    
                    self.logger.info(
                        f"Order created - ID: {order.order_id}, Symbol: {order.symbol}, "
                        f"Side: {order.side.value}, Quantity: {order.quantity}"
                    )
                else:
                    self.logger.warning(
                        f"Signal rejected - Signal: {signal}, Reason: Failed risk checks"
                    )
                    
            except Exception as e:
                self.logger.error(
                    f"Signal processing error - Signal: {signal}, Error: {str(e)}"
                )
        
        return orders
    
    def get_portfolio_state(self) -> PortfolioStateProtocol:
        """Get current portfolio state."""
        return self._resolver.get_portfolio_state()
    
    def update_fills(self, fills: List[Dict[str, Any]]) -> None:
        """Update portfolio with executed fills."""
        portfolio_state = self._resolver.get_portfolio_state()
        
        for fill in fills:
            try:
                # Update portfolio state
                quantity_delta = fill["quantity"]
                if fill["side"] == "sell":
                    quantity_delta = -quantity_delta
                
                position = portfolio_state.update_position(
                    symbol=fill["symbol"],
                    quantity_delta=quantity_delta,
                    price=fill["price"],
                    timestamp=fill["timestamp"]
                )
                
                # Update cash for commission
                commission = fill.get("commission", Decimal(0))
                if commission:
                    portfolio_state._cash_balance -= commission
                
                # Move order to history
                order_id = fill.get("order_id")
                if order_id and order_id in self._active_orders:
                    order = self._active_orders.pop(order_id)
                    self._order_history.append(order)
                
                # Emit fill event
                self._emit_event(
                    EventType.FILL,
                    {
                        "type": "fill_processed",
                        "fill": fill,
                        "position": position
                    }
                )
                
                self.logger.info(
                    f"Fill processed - Symbol: {fill['symbol']}, Side: {fill['side']}, "
                    f"Quantity: {fill['quantity']}, Price: {fill['price']}"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Fill processing error - Fill: {fill}, Error: {str(e)}"
                )
    
    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        """Update market data for risk calculations."""
        prices = market_data.get("prices", {})
        if prices:
            decimal_prices = {
                symbol: Decimal(str(price))
                for symbol, price in prices.items()
            }
            portfolio_state = self._resolver.get_portfolio_state()
            portfolio_state.update_market_prices(decimal_prices)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        portfolio_state = self._resolver.get_portfolio_state()
        metrics = portfolio_state.get_risk_metrics()
        risk_limits = self._resolver.get_risk_limits()
        
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
                for symbol, pos in portfolio_state.get_all_positions().items()
            },
            "risk_limits": [
                {
                    "type": type(limit).__name__,
                    "info": limit.get_limit_info()
                }
                for limit in risk_limits
            ],
            "active_strategies": list(self._strategies.keys()),
            "active_orders": len(self._active_orders),
            "order_history_count": len(self._order_history)
        }
    
    def _setup_portfolio_state(
        self,
        initial_capital: Decimal,
        base_currency: str
    ) -> None:
        """Setup portfolio state with dependency injection."""
        portfolio_state = PortfolioState(
            initial_capital=initial_capital,
            base_currency=base_currency
        )
        self._resolver.register_portfolio_state(portfolio_state)
    
    def _emit_event(self, event_type: EventType, data: Dict[str, Any]) -> None:
        """Emit event through event system."""
        if self.event_bus:
            event = Event(
                event_type=event_type,
                source_id=self._component_id,
                payload=data
            )
            self.event_bus.publish(event)
        else:
            # Log event if no event system available
            self.logger.debug(
                f"Event emitted - Type: {event_type.value}, Data: {data}"
            )


def create_risk_portfolio_container(
    component_id: str,
    dependency_container: DependencyContainer,
    initial_capital: Decimal = Decimal("100000"),
    base_currency: str = "USD"
) -> RiskPortfolioContainer:
    """Factory function to create a properly configured risk portfolio container.
    
    Args:
        component_id: Unique component identifier
        dependency_container: DI container
        initial_capital: Starting capital
        base_currency: Base currency
        
    Returns:
        Configured risk portfolio container
    """
    container = RiskPortfolioContainer(
        component_id=component_id,
        dependency_container=dependency_container,
        initial_capital=initial_capital,
        base_currency=base_currency
    )
    
    # Register in dependency container
    dependency_container.register_instance(
        'RiskPortfolio',
        container,
        metadata={'component_type': 'risk_portfolio'}
    )
    
    return container
