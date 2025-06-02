"""
Fixed execution containers with proper event routing to prevent cycles and duplicates.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
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


class StrategyContainer(BaseComposableContainer):
    """Fixed container for strategy execution with proper signal routing."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Configure external communication via Event Router
        self._configure_external_events(config)
        
        super().__init__(
            role=ContainerRole.STRATEGY,
            name="StrategyContainer",
            config=config,
            container_id=container_id
        )
        # Strategy execution components
        self.strategy: Optional[Strategy] = None
        self.signal_aggregator = None
        
        # State management
        self._current_indicators: Dict[str, Any] = {}
        self._current_market_data: Dict[str, Any] = {}
        
        # Deduplication tracking
        self._processed_signals: Set[str] = set()
        self._signal_cleanup_interval = 300  # 5 minutes
        
        # Determine if this is multi-strategy
        self.strategies_config = self._get_strategies_config()
        self.is_multi_strategy = len(self.strategies_config) > 1
    
    def _configure_external_events(self, config: Dict[str, Any]) -> None:
        """Configure external Event Router communication"""
        if 'external_events' not in config:
            config['external_events'] = {}
        
        ext_config = config['external_events']
        
        # Configure publications - ONLY for sub-containers in multi-strategy
        if 'publishes' not in ext_config:
            ext_config['publishes'] = []
        
        # Check if this is a sub-container
        is_sub_container = '_' in config.get('container_id', '')
        
        if is_sub_container:
            # Sub-containers publish to parent only
            ext_config['publishes'].extend([
                {
                    'events': ['SIGNAL'],
                    'scope': 'PARENT',  # Only to parent container
                    'tier': 'standard'
                }
            ])
        else:
            # Main strategy container doesn't publish externally if it has a parent
            # It uses internal event bus to communicate with parent
            pass
        
        # Configure subscriptions
        if 'subscribes' not in ext_config:
            ext_config['subscribes'] = []
        
        ext_config['subscribes'].extend([
            {
                'source': '*',  # Subscribe to BAR events from any source
                'events': ['BAR'],
                'tier': 'fast',
                'filters': {}
            },
            {
                'source': '*',  # Subscribe to INDICATORS events from any source  
                'events': ['INDICATORS'],
                'tier': 'standard',
                'filters': {'subscriber': config.get('container_id', 'strategy_container')}
            }
        ])
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process events and generate signals with deduplication."""
        await super().process_event(event)
        
        logger.info(f"StrategyContainer received event: {event.event_type}")
        
        if self.is_multi_strategy:
            # Multi-strategy: forward events to sub-containers and collect signals
            if event.event_type == EventType.SIGNAL:
                # Check if this is from a sub-container
                source = event.payload.get('source', '')
                if source.startswith(self.metadata.container_id + '_'):
                    # Signal from sub-container - aggregate and forward
                    await self._handle_sub_container_signal(event)
            else:
                # Forward other events to all sub-containers
                for child in self.child_containers:
                    await child.process_event(event)
        else:
            # Single strategy: handle events directly
            if event.event_type == EventType.INDICATORS:
                await self._handle_indicators_event(event)
            elif event.event_type == EventType.BAR:
                await self._handle_bar_event(event)
        
        return None
    
    async def _handle_sub_container_signal(self, event: Event) -> None:
        """Collect and aggregate signals from sub-containers with deduplication."""
        signals = event.payload.get('signals', [])
        if signals:
            # Create unique signal IDs for deduplication
            new_signals = []
            for signal in signals:
                signal_id = f"{signal.symbol}_{signal.side}_{signal.strength}_{event.timestamp}"
                if signal_id not in self._processed_signals:
                    self._processed_signals.add(signal_id)
                    new_signals.append(signal)
                else:
                    logger.debug(f"Duplicate signal filtered: {signal_id}")
            
            if new_signals:
                logger.info(f"Multi-strategy container forwarding {len(new_signals)} unique signals")
                
                # Forward to parent using INTERNAL event bus only
                if self.parent_container:
                    forward_event = Event(
                        event_type=EventType.SIGNAL,
                        payload={
                            'timestamp': event.timestamp,
                            'signals': new_signals,
                            'market_data': event.payload.get('market_data', {}),
                            'source': self.metadata.container_id,
                            'container_type': 'strategy_aggregator'
                        },
                        timestamp=event.timestamp
                    )
                    # Use internal bus to parent
                    self.publish_event(forward_event, target_scope="parent")
    
    async def _emit_signals(self, signals: List, timestamp, market_data: Dict[str, Any]) -> None:
        """Emit signals to parent container with proper routing."""
        if signals:
            for signal in signals:
                log_signal_event(logger, signal)
            
            signal_event = Event(
                event_type=EventType.SIGNAL,
                payload={
                    'timestamp': timestamp,
                    'signals': signals,
                    'market_data': market_data,
                    'source': self.metadata.container_id,
                    'container_type': 'strategy'
                },
                timestamp=timestamp
            )
            
            # ALWAYS use internal communication to parent
            if self.parent_container:
                logger.info(f"🚀 StrategyContainer publishing SIGNAL to parent via internal bus")
                self.publish_event(signal_event, target_scope="parent")
            else:
                # Only use external if truly orphaned (shouldn't happen)
                logger.warning(f"StrategyContainer has no parent, using external routing")
                from ..core.events.hybrid_interface import CommunicationTier
                self.publish_external(signal_event, tier=CommunicationTier.STANDARD)
    
    # ... rest of methods remain the same ...


class PortfolioContainer(BaseComposableContainer):
    """Fixed container for portfolio allocation with proper signal routing."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Configure external communication - REMOVED external SIGNAL publishing
        if 'external_events' not in config:
            config['external_events'] = {}
        
        # Portfolio container only subscribes externally, doesn't publish signals externally
        config['external_events']['subscribes'] = config['external_events'].get('subscribes', []) + [
            {
                'source': '*',
                'events': ['BAR', 'INDICATORS'],
                'tier': 'fast'
            }
        ]
        
        super().__init__(
            role=ContainerRole.PORTFOLIO,
            name="PortfolioContainer",
            config=config,
            container_id=container_id
        )
        self.allocation_manager = None
        self._processed_signals: Set[str] = set()
        
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process signals with deduplication."""
        await super().process_event(event)
        
        # Forward BAR and INDICATOR events to children (StrategyContainer)
        if event.event_type in [EventType.BAR, EventType.INDICATORS]:
            logger.debug(f"PortfolioContainer forwarding {event.event_type} to children")
            self.publish_event(event, target_scope="children")
        
        if event.event_type == EventType.SIGNAL and self.allocation_manager:
            signals = event.payload.get('signals', [])
            logger.info(f"📊 PortfolioContainer received SIGNAL event with {len(signals)} signals")
            
            # Deduplicate signals
            unique_signals = []
            for signal in signals:
                signal_id = f"{signal.symbol}_{signal.side}_{signal.strength}_{event.timestamp}"
                if signal_id not in self._processed_signals:
                    self._processed_signals.add(signal_id)
                    unique_signals.append(signal)
            
            if unique_signals:
                allocated_event = Event(
                    event_type=EventType.SIGNAL,
                    payload={
                        'timestamp': event.timestamp,
                        'signals': unique_signals,
                        'market_data': event.payload.get('market_data', {}),
                        'source': self.metadata.container_id
                    },
                    timestamp=event.timestamp
                )
                
                # Forward to parent (risk container) via INTERNAL bus only
                if self.parent_container:
                    logger.info(f"📤 PortfolioContainer forwarding {len(unique_signals)} signals to parent")
                    self.publish_event(allocated_event, target_scope="parent")
                
                return allocated_event
        
        return None


class RiskContainer(BaseComposableContainer):
    """Fixed container for risk management with deduplication."""
    
    def __init__(self, config: Dict[str, Any], container_id: str = None):
        # Configure external communication
        if 'external_events' not in config:
            config['external_events'] = {}
        
        # RiskContainer only publishes ORDERs externally, not subscribes to SIGNALs
        config['external_events']['publishes'] = config['external_events'].get('publishes', []) + [
            {
                'events': ['ORDER'],
                'scope': 'GLOBAL',
                'tier': 'standard'
            }
        ]
        
        # Remove external SIGNAL subscription - signals come via internal bus
        config['external_events']['subscribes'] = config['external_events'].get('subscribes', []) + [
            {
                'source': '*',
                'events': ['FILL'],
                'tier': 'standard'
            }
        ]
        
        super().__init__(
            role=ContainerRole.RISK,
            name="RiskContainer",
            config=config,
            container_id=container_id
        )
        self.risk_manager = None
        self._processed_signals: Set[str] = set()
        self._signal_window = {}  # Track signals by timestamp for deduplication
        
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process signals with deduplication and proper order generation."""
        await super().process_event(event)
        
        logger.debug(f"🔍 RiskContainer process_event called with: {event.event_type}")
        
        # Forward BAR events to children for market data
        if event.event_type == EventType.BAR:
            logger.debug(f"RiskContainer forwarding BAR event to children")
            self.publish_event(event, target_scope="children")
        
        elif event.event_type == EventType.FILL and self.risk_manager:
            # Process fill events to update portfolio
            fills = event.payload.get('fills', [])
            logger.info(f"🎯 RiskContainer received FILL event with {len(fills)} fills")
            
            for fill in fills:
                self.risk_manager.update_fills([fill])
                logger.info(f"📊 Portfolio updated with fill: {fill.fill_id}")
            
            # Log portfolio state
            portfolio_state = self.risk_manager.get_portfolio_state()
            cash_balance = portfolio_state.get_cash_balance()
            positions = portfolio_state.get_all_positions()
            
            logger.info(f"💼 Portfolio Update - Cash: ${cash_balance:.2f}, Positions: {len(positions)}")
            for symbol, position in positions.items():
                if position.quantity != 0:
                    logger.info(f"   📈 Position: {symbol} {position.quantity} shares @ ${position.average_price:.2f}")
        
        elif event.event_type == EventType.SIGNAL and self.risk_manager:
            signals = event.payload.get('signals', [])
            market_data = event.payload.get('market_data', {})
            
            # Deduplicate signals within time window
            current_time = event.timestamp
            unique_signals = []
            
            for signal in signals:
                # Create unique signal identifier
                signal_id = f"{signal.symbol}_{signal.side}_{signal.strength}"
                signal_window_key = f"{signal_id}_{current_time}"
                
                # Check if we've seen this signal recently (within 1 second)
                recent_key = None
                for key in list(self._signal_window.keys()):
                    if key.startswith(signal_id):
                        stored_time = self._signal_window[key]
                        if abs((current_time - stored_time).total_seconds()) < 1.0:
                            recent_key = key
                            break
                
                if not recent_key:
                    # New unique signal
                    self._signal_window[signal_window_key] = current_time
                    unique_signals.append(signal)
                    logger.info(f"🔥 RiskContainer processing unique signal: {signal.symbol} {signal.side}")
                else:
                    logger.debug(f"Duplicate signal filtered: {signal_id}")
            
            # Clean old entries from signal window (older than 5 seconds)
            for key in list(self._signal_window.keys()):
                if (current_time - self._signal_window[key]).total_seconds() > 5:
                    del self._signal_window[key]
            
            if unique_signals:
                logger.info(f"🔥 RiskContainer processing {len(unique_signals)} unique signals")
                
                # Transform market data for position sizer
                transformed_market_data = {
                    "prices": {
                        symbol: data['close'] if isinstance(data, dict) else data
                        for symbol, data in market_data.items()
                    }
                }
                
                # Process signals through risk management
                orders = self.risk_manager.process_signals(unique_signals, transformed_market_data)
                
                logger.info(f"📋 RiskContainer generated {len(orders)} orders")
                
                if orders:
                    order_event = Event(
                        event_type=EventType.ORDER,
                        payload={
                            'timestamp': event.timestamp,
                            'orders': orders,
                            'market_data': market_data,  # Include original market data
                            'source': self.metadata.container_id
                        },
                        timestamp=event.timestamp
                    )
                    
                    # Publish ORDER event externally to ExecutionContainer
                    logger.info(f"📤 RiskContainer publishing ORDER event via Event Router")
                    self.publish_external(order_event)
                    
                    return order_event
        
        return None