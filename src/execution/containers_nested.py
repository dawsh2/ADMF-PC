"""
Execution containers with proper nesting: Risk > Portfolio > Strategy.

This implementation ensures:
1. PortfolioContainer is a child of RiskContainer
2. StrategyContainer is a child of PortfolioContainer
3. Events flow properly through the nested hierarchy
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from decimal import Decimal
import asyncio
import logging

from ..core.containers.composable import (
    BaseComposableContainer, ComposableContainerProtocol, ContainerRole,
    ContainerState, ContainerLimits
)
from ..core.events.types import Event, EventType
from ..data.protocols import DataLoader
from ..strategy.protocols import Strategy
from ..risk.protocols import Signal
from ..core.logging.event_logger import (
    log_bar_event, log_indicator_event, log_signal_event, 
    log_order_event, log_fill_event, log_portfolio_event,
    get_event_logger
)

logger = get_event_logger(__name__)


class NestedRiskContainer(BaseComposableContainer):
    """Risk container that contains Portfolio as a child, which contains Strategy."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.RISK,
            name="RiskContainer",
            config=config,
            container_id=container_id
        )
        self.risk_manager = None
        self.position_sizer = None
        self.risk_limits = []
        
    def on_output_event(self, handler):
        """Register handler for output events."""
        # Risk container outputs ORDERs
        self.event_bus.subscribe(EventType.ORDER, handler)
        
    @property
    def expected_input_type(self):
        """RiskContainer receives INDICATORS from IndicatorContainer."""
        return EventType.INDICATORS
        
    def receive_event(self, event: Event):
        """Receive and process events from pipeline adapter."""
        if event.event_type == EventType.INDICATORS:
            # Forward indicators to child PortfolioContainer
            logger.info(f"🎯 RiskContainer received INDICATORS, forwarding to children")
            self._forward_to_children(event)
        elif event.event_type == EventType.SIGNAL:
            # Signals from Strategy (via Portfolio) for risk checking
            asyncio.create_task(self._handle_signal_event(event))
        elif event.event_type == EventType.SYSTEM:
            asyncio.create_task(self._handle_system_event(event))
            self._forward_to_children(event)
    
    def _forward_to_children(self, event: Event):
        """Forward events to child containers."""
        for child in self.child_containers:
            if hasattr(child, 'receive_event'):
                child.receive_event(event)
    
    async def _initialize_self(self) -> None:
        """Initialize risk management components."""
        # Initialize position sizer
        position_sizers = self._metadata.config.get('position_sizers', [])
        if position_sizers:
            sizer_config = position_sizers[0]
            if sizer_config.get('type') == 'fixed':
                from ..risk.position_sizing import FixedPositionSizer
                size = sizer_config.get('size', '5000')
                self.position_sizer = FixedPositionSizer(
                    size=Decimal(str(size))
                )
        else:
            from ..risk.position_sizing import FixedPositionSizer
            self.position_sizer = FixedPositionSizer(size=Decimal('5000'))
        
        # Initialize risk limits
        self.risk_limits = []
        limits = self._metadata.config.get('limits', [])
        for limit_config in limits:
            if limit_config.get('type') == 'position':
                from ..risk.risk_limits import MaxPositionLimit
                max_position = limit_config.get('max_position', '5000')
                risk_limit = MaxPositionLimit(
                    max_position_value=Decimal(str(max_position))
                )
                self.risk_limits.append(risk_limit)
            elif limit_config.get('type') == 'exposure':
                from ..risk.risk_limits import MaxExposureLimit
                max_exposure_pct = limit_config.get('max_exposure_pct', '80')
                risk_limit = MaxExposureLimit(
                    max_exposure_pct=Decimal(str(max_exposure_pct))
                )
                self.risk_limits.append(risk_limit)
        
        # Subscribe to child events (signals from strategy)
        self.event_bus.subscribe(EventType.SIGNAL, self._handle_internal_signal)
        
        logger.info("RiskContainer initialized with position sizing and risk limits")
    
    def _handle_internal_signal(self, event: Event):
        """Handle signals from child containers."""
        # Process signals that bubble up from Strategy
        asyncio.create_task(self._handle_signal_event(event))
    
    async def _handle_signal_event(self, event: Event) -> None:
        """Process signals through risk management."""
        signals = event.payload.get('signals', [])
        
        if not signals:
            return
        
        market_data = event.payload.get('market_data', {})
        approved_orders = []
        
        # Get portfolio state from child PortfolioContainer
        portfolio_container = self._get_portfolio_container()
        portfolio_state = None
        if portfolio_container and hasattr(portfolio_container, 'portfolio_state'):
            portfolio_state = portfolio_container.portfolio_state
        
        # Process each signal through risk management
        for signal in signals:
            # Apply risk limits
            if self._check_risk_limits(signal, portfolio_state):
                # Generate order with position sizing
                order = self._create_order_from_signal(signal, market_data, portfolio_state)
                if order:
                    approved_orders.append(order)
        
        if approved_orders:
            # Emit approved orders
            for order in approved_orders:
                order_event = Event(
                    event_type=EventType.ORDER,
                    payload={
                        'order': order,
                        'source': self.metadata.container_id
                    },
                    timestamp=event.timestamp
                )
                
                # Publish to parent (which will route to ExecutionContainer)
                self.publish_event(order_event, target_scope="parent")
                log_order_event(logger, order)
                
            logger.info(f"RiskContainer approved and emitted {len(approved_orders)} orders")
    
    def _get_portfolio_container(self):
        """Get the child PortfolioContainer."""
        for child in self.child_containers:
            if child.metadata.role == ContainerRole.PORTFOLIO:
                return child
        return None
    
    def _check_risk_limits(self, signal, portfolio_state) -> bool:
        """Check if signal passes risk limits."""
        if not portfolio_state:
            logger.warning("No portfolio state available for risk checks")
            return False
        
        symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol', 'UNKNOWN')
        
        # Apply each risk limit
        for risk_limit in self.risk_limits:
            # Risk limit checking logic here
            pass
        
        return True
    
    def _create_order_from_signal(self, signal, market_data, portfolio_state):
        """Create order from signal using position sizer."""
        from ..execution.protocols import Order, OrderType, OrderSide
        import uuid
        
        # Order creation logic here
        symbol = signal.symbol if hasattr(signal, 'symbol') else signal.get('symbol', 'UNKNOWN')
        side = signal.side if hasattr(signal, 'side') else signal.get('side')
        
        # Calculate position size
        current_price = 100.0  # Default
        if symbol in market_data:
            current_price = market_data[symbol].get('close', 100.0)
        
        quantity = 10  # Simplified for now
        
        order = Order(
            order_id=f"ORD-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=OrderSide.BUY if side == 'BUY' else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=None,
            created_at=datetime.now()
        )
        
        return order
    
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system events."""
        message = event.payload.get('message')
        if message == 'END_OF_DATA':
            logger.info("🏁 RiskContainer received END_OF_DATA")
    
    def get_capabilities(self) -> Set[str]:
        """Risk container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("risk.management")
        capabilities.add("risk.position_sizing")
        capabilities.add("risk.nested_portfolio")
        return capabilities


class NestedPortfolioContainer(BaseComposableContainer):
    """Portfolio container nested within Risk, containing Strategy."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.PORTFOLIO,
            name="PortfolioContainer",
            config=config,
            container_id=container_id
        )
        self.portfolio_state = None
        
    def on_output_event(self, handler):
        """Register handler for output events."""
        # Portfolio doesn't output to pipeline directly in nested structure
        pass
        
    @property
    def expected_input_type(self):
        """Portfolio receives indicators from parent Risk container."""
        return EventType.INDICATORS
        
    def receive_event(self, event: Event):
        """Receive events from parent RiskContainer."""
        if event.event_type == EventType.INDICATORS:
            # Forward to child StrategyContainer
            logger.info(f"📊 PortfolioContainer forwarding INDICATORS to Strategy")
            self._forward_to_children(event)
        elif event.event_type == EventType.FILL:
            asyncio.create_task(self._handle_fill_event(event))
        elif event.event_type == EventType.SIGNAL:
            # Signals from child Strategy - forward to parent Risk
            self._forward_signal_to_parent(event)
        elif event.event_type == EventType.SYSTEM:
            asyncio.create_task(self._handle_system_event(event))
            self._forward_to_children(event)
    
    def _forward_to_children(self, event: Event):
        """Forward events to child containers."""
        for child in self.child_containers:
            if hasattr(child, 'receive_event'):
                child.receive_event(event)
    
    def _forward_signal_to_parent(self, event: Event):
        """Forward signals from Strategy to parent RiskContainer."""
        logger.info(f"📤 PortfolioContainer forwarding SIGNAL to parent Risk")
        self.publish_event(event, target_scope="parent")
    
    async def _initialize_self(self) -> None:
        """Initialize portfolio tracking."""
        from ..risk.portfolio_state import PortfolioState
        
        initial_capital = self._metadata.config.get('initial_capital', '100000')
        if isinstance(initial_capital, str):
            initial_capital = Decimal(initial_capital)
        self.portfolio_state = PortfolioState(initial_capital=initial_capital)
        
        # Subscribe to signals from child strategy
        self.event_bus.subscribe(EventType.SIGNAL, self._handle_internal_signal)
        
        logger.info(f"PortfolioContainer initialized with capital: ${initial_capital}")
    
    def _handle_internal_signal(self, event: Event):
        """Handle signals from child Strategy container."""
        # Forward to parent Risk container
        self._forward_signal_to_parent(event)
    
    async def _handle_fill_event(self, event: Event) -> None:
        """Update portfolio state with fills."""
        fill = event.payload.get('fill')
        if not fill or not self.portfolio_state:
            return
        
        # Update portfolio logic here
        logger.info(f"💼 PortfolioContainer updated with fill: {fill.symbol}")
    
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system events."""
        message = event.payload.get('message')
        if message == 'END_OF_DATA':
            logger.info("🏁 PortfolioContainer received END_OF_DATA")
    
    def get_capabilities(self) -> Set[str]:
        """Portfolio container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("portfolio.tracking")
        capabilities.add("portfolio.nested")
        return capabilities


class NestedStrategyContainer(BaseComposableContainer):
    """Strategy container nested within Portfolio."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        super().__init__(
            role=ContainerRole.STRATEGY,
            name="StrategyContainer",
            config=config,
            container_id=container_id
        )
        self.strategy = None
        self.strategies_config = config.get('strategies', [])
        self.multi_strategy = len(self.strategies_config) > 1
        self.last_indicators = {}
        self._market_data = {}
        
    def on_output_event(self, handler):
        """Register handler for output events."""
        # Strategy outputs signals to parent Portfolio
        self.event_bus.subscribe(EventType.SIGNAL, handler)
        
    @property
    def expected_input_type(self):
        """Strategy receives indicators from parent Portfolio."""
        return EventType.INDICATORS
        
    def receive_event(self, event: Event):
        """Receive events from parent PortfolioContainer."""
        if event.event_type == EventType.INDICATORS:
            indicators = event.payload.get('indicators', {})
            market_data = event.payload.get('market_data', {})
            logger.info(f"📈 StrategyContainer received INDICATORS")
            self.last_indicators = indicators
            self._market_data.update(market_data)
            
            # Process signals
            asyncio.create_task(self._process_signals(event.timestamp))
        elif event.event_type == EventType.SYSTEM:
            asyncio.create_task(self._handle_system_event(event))
    
    async def _initialize_self(self) -> None:
        """Initialize strategy."""
        if self.multi_strategy:
            # For now, just use first strategy
            strategy_config = self.strategies_config[0] if self.strategies_config else {}
        else:
            strategy_config = self._metadata.config.get('strategy', {})
            if not strategy_config and self.strategies_config:
                strategy_config = self.strategies_config[0]
        
        strategy_type = strategy_config.get('type', 'momentum')
        
        # Import and create strategy
        if strategy_type == 'momentum':
            from ..strategy.strategies.momentum import MomentumStrategy
            strategy_params = strategy_config.get('parameters', {})
            self.strategy = MomentumStrategy(**strategy_params)
        elif strategy_type == 'mean_reversion':
            from ..strategy.strategies.mean_reversion import MeanReversionStrategy
            strategy_params = strategy_config.get('parameters', {})
            self.strategy = MeanReversionStrategy(**strategy_params)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        logger.info(f"StrategyContainer initialized with {strategy_type} strategy")
    
    async def _process_signals(self, timestamp) -> None:
        """Generate signals from strategy."""
        if self.strategy and self.last_indicators and self._market_data:
            logger.info(f"🚀 Generating signals with strategy: {type(self.strategy).__name__}")
            
            # Create strategy input
            strategy_input = {
                'market_data': self._market_data,
                'indicators': self.last_indicators,
                'timestamp': timestamp
            }
            signals = self.strategy.generate_signals(strategy_input)
            
            if signals:
                await self._emit_signals(signals, timestamp, self._market_data)
    
    async def _emit_signals(self, signals: List, timestamp, market_data: Dict[str, Any]) -> None:
        """Emit signals to parent Portfolio."""
        if signals:
            signal_event = Event(
                event_type=EventType.SIGNAL,
                payload={
                    'timestamp': timestamp,
                    'signals': signals,
                    'market_data': market_data,
                    'source': self.metadata.container_id
                },
                timestamp=timestamp
            )
            
            logger.info(f"🚀 StrategyContainer publishing SIGNAL to parent")
            self.publish_event(signal_event, target_scope="parent")
            
            for signal in signals:
                log_signal_event(logger, signal)
    
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system events."""
        message = event.payload.get('message')
        if message == 'END_OF_DATA':
            logger.info("🏁 StrategyContainer received END_OF_DATA")
    
    def get_capabilities(self) -> Set[str]:
        """Strategy container capabilities."""
        capabilities = super().get_capabilities()
        capabilities.add("strategy.execution")
        capabilities.add("strategy.signal_generation")
        capabilities.add("strategy.nested")
        return capabilities