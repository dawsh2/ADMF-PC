# Semantic Events Implementation Guide for ADMF-PC

# ============================================
# OVERVIEW
# ============================================
"""
This guide shows how to integrate semantic events into your existing event system.
Semantic events provide type safety and domain modeling while maintaining your
Protocol + Composition architecture.

The integration is ADDITIVE - your existing Event class continues to work,
but you gain the option to use strongly-typed semantic events where it makes sense.
"""

# ============================================
# STEP 1: Add Semantic Event Files to Your Structure
# ============================================
"""
Add these files to your existing events module:

src/core/events/
├── __init__.py          (update imports)
├── protocols.py         (add SemanticEventProtocol)
├── bus.py              (no changes needed!)
├── types.py            (keep existing Event class)
├── semantic/           (NEW directory)
│   ├── __init__.py
│   ├── protocols.py    (semantic event protocol)
│   ├── base.py         (helper functions)
│   ├── trading.py      (trading-specific events)
│   ├── market.py       (market data events)
│   └── validation.py   (validation utilities)
└── observers/
    └── tracer.py       (update to handle semantic events)
"""

# ============================================
# File: src/core/events/semantic/protocols.py
# ============================================
"""Semantic event protocols."""

from typing import Protocol, runtime_checkable, Optional, Dict, Any
from datetime import datetime

@runtime_checkable
class SemanticEventProtocol(Protocol):
    """Protocol defining what makes something a semantic event.
    
    Any object with these fields IS a semantic event.
    No inheritance required!
    """
    
    # Identity
    event_id: str
    schema_version: str
    timestamp: datetime
    
    # Correlation tracking
    correlation_id: str
    causation_id: Optional[str]
    
    # Source information
    source_container: str
    source_component: str
    
    # Business context
    strategy_id: Optional[str]
    portfolio_id: Optional[str]
    
    def validate(self) -> bool:
        """Validate the event."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        ...
    
    def to_legacy_event(self) -> 'Event':
        """Convert to legacy Event for compatibility."""
        ...


# ============================================
# File: src/core/events/semantic/base.py
# ============================================
"""Base utilities for semantic events."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, TypeVar, Type
from datetime import datetime
import uuid

from ..types import Event, EventType

# Type variable for generic functions
T = TypeVar('T')

def make_event_id() -> str:
    """Generate unique event ID."""
    return f"evt_{uuid.uuid4().hex[:12]}"

def make_correlation_id() -> str:
    """Generate correlation ID for event flow."""
    return f"corr_{uuid.uuid4().hex[:12]}"

@dataclass
class SemanticEventBase:
    """Base fields for semantic events.
    
    Use this as a mixin with your event dataclasses:
    
    @dataclass
    class MyEvent(SemanticEventBase):
        my_field: str = ""
    """
    
    # Identity
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Correlation tracking
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None
    
    # Source information
    source_container: str = ""
    source_component: str = ""
    
    # Business context
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        # Add type information
        d['__type__'] = self.__class__.__name__
        # Convert datetime to ISO format
        if isinstance(d.get('timestamp'), datetime):
            d['timestamp'] = d['timestamp'].isoformat()
        return d
    
    def to_legacy_event(self) -> Event:
        """Convert to legacy Event for compatibility."""
        # Map semantic event type to EventType enum
        event_type_map = {
            'PositionOpenEvent': EventType.POSITION_OPEN,
            'PositionCloseEvent': EventType.POSITION_CLOSE,
            'TradingSignal': EventType.SIGNAL,
            'OrderEvent': EventType.ORDER,
            'FillEvent': EventType.FILL,
            'MarketDataEvent': EventType.BAR,
        }
        
        event_type = event_type_map.get(
            self.__class__.__name__, 
            EventType.SIGNAL  # Default
        )
        
        return Event(
            event_type=event_type.value,
            payload=self.to_dict(),
            source_id=self.source_container,
            container_id=self.source_container,
            correlation_id=self.correlation_id,
            causation_id=self.causation_id,
            timestamp=self.timestamp,
            metadata={
                'semantic': True,
                'schema_version': self.schema_version,
                'event_id': self.event_id
            }
        )

def create_event_with_context(
    event_class: Type[T],
    container_id: str,
    component: str = "",
    **kwargs
) -> T:
    """Create semantic event with container context."""
    return event_class(
        source_container=container_id,
        source_component=component,
        **kwargs
    )

def create_caused_event(
    causing_event: Any,
    event_class: Type[T],
    **kwargs
) -> T:
    """Create event caused by another event."""
    # Preserve correlation
    correlation_id = causing_event.correlation_id
    if hasattr(causing_event, 'event_id'):
        causation_id = causing_event.event_id
    else:
        causation_id = causing_event.metadata.get('event_id')
    
    return event_class(
        correlation_id=correlation_id,
        causation_id=causation_id,
        **kwargs
    )


# ============================================
# File: src/core/events/semantic/trading.py
# ============================================
"""Trading-specific semantic events."""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from datetime import datetime

from .base import SemanticEventBase

@dataclass
class PositionOpenEvent(SemanticEventBase):
    """Event for opening a position."""
    
    # Position details
    symbol: str = ""
    quantity: int = 0
    side: Literal["LONG", "SHORT"] = "LONG"
    entry_price: float = 0.0
    
    # Risk parameters
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_position_pct: float = 0.02
    
    def validate(self) -> bool:
        """Validate position open event."""
        return (
            self.symbol != "" and
            self.quantity > 0 and
            self.entry_price > 0 and
            0 < self.max_position_pct <= 1.0
        )

@dataclass
class PositionCloseEvent(SemanticEventBase):
    """Event for closing a position."""
    
    # Position details
    symbol: str = ""
    quantity: int = 0
    side: Literal["LONG", "SHORT"] = "LONG"
    entry_price: float = 0.0
    exit_price: float = 0.0
    
    # Results
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_seconds: Optional[float] = None
    
    # Reason for close
    close_reason: Literal["SIGNAL", "STOP_LOSS", "TAKE_PROFIT", "EOD", "RISK"] = "SIGNAL"
    
    def validate(self) -> bool:
        """Validate position close event."""
        return (
            self.symbol != "" and
            self.quantity > 0 and
            self.entry_price > 0 and
            self.exit_price > 0
        )

@dataclass
class TradingSignal(SemanticEventBase):
    """Trading signal event."""
    
    schema_version: str = "2.0.0"  # Override version
    
    # Signal details
    symbol: str = ""
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    strength: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.5  # 0.0 to 1.0
    
    # Trading parameters
    suggested_size: Optional[float] = None
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    time_horizon: Optional[str] = None  # "intraday", "swing", "position"
    
    # Context
    regime_context: Optional[str] = None
    indicators_used: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate trading signal."""
        return (
            self.symbol != "" and
            0.0 <= self.strength <= 1.0 and
            0.0 <= self.confidence <= 1.0
        )

@dataclass
class OrderEvent(SemanticEventBase):
    """Order placement event."""
    
    # Order details
    order_id: str = field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:8]}")
    symbol: str = ""
    side: Literal["BUY", "SELL"] = "BUY"
    quantity: int = 0
    order_type: Literal["MARKET", "LIMIT", "STOP", "STOP_LIMIT"] = "MARKET"
    
    # Pricing
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Risk parameters
    time_in_force: Literal["DAY", "GTC", "IOC", "FOK"] = "DAY"
    reduce_only: bool = False
    
    def validate(self) -> bool:
        """Validate order event."""
        valid = self.symbol != "" and self.quantity > 0
        
        if self.order_type in ("LIMIT", "STOP_LIMIT") and self.limit_price is None:
            return False
        if self.order_type in ("STOP", "STOP_LIMIT") and self.stop_price is None:
            return False
        
        return valid

@dataclass
class FillEvent(SemanticEventBase):
    """Order fill event."""
    
    # Fill details
    fill_id: str = field(default_factory=lambda: f"fill_{uuid.uuid4().hex[:8]}")
    order_id: str = ""
    symbol: str = ""
    side: Literal["BUY", "SELL"] = "BUY"
    
    # Execution details
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    # Venue information
    venue: Optional[str] = None
    liquidity_type: Optional[Literal["MAKER", "TAKER"]] = None
    
    def validate(self) -> bool:
        """Validate fill event."""
        return (
            self.order_id != "" and
            self.symbol != "" and
            self.quantity > 0 and
            self.price > 0
        )


# ============================================
# File: src/core/events/semantic/validation.py
# ============================================
"""Validation utilities for semantic events."""

from typing import Any, List, Tuple, Optional
import inspect

from .protocols import SemanticEventProtocol

def validate_semantic_event(event: Any) -> Tuple[bool, List[str]]:
    """Validate any object claiming to be a semantic event.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if it implements the protocol
    if not isinstance(event, SemanticEventProtocol):
        # Check required fields manually
        required_fields = [
            'event_id', 'schema_version', 'timestamp',
            'correlation_id', 'source_container'
        ]
        
        for field in required_fields:
            if not hasattr(event, field):
                errors.append(f"Missing required field: {field}")
    
    # Check if it has validate method
    if hasattr(event, 'validate'):
        try:
            if not event.validate():
                errors.append("Event validation failed")
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
    
    # Validate correlation ID
    if hasattr(event, 'correlation_id') and not event.correlation_id:
        errors.append("Missing correlation_id")
    
    return len(errors) == 0, errors

def is_semantic_event(obj: Any) -> bool:
    """Check if object is a semantic event."""
    return isinstance(obj, SemanticEventProtocol)


# ============================================
# STEP 2: Update Your Event Bus to Handle Both
# ============================================
"""
Your EventBus doesn't need changes! It already handles any object.
But here's how containers can work with semantic events:
"""

# Example container using semantic events
class PortfolioContainer:
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.event_bus = EventBus(container_id)
        
        # Subscribe to both legacy and semantic events
        self.event_bus.subscribe(EventType.POSITION_OPEN, self.on_position_open)
        self.event_bus.subscribe(EventType.FILL, self.on_fill)
    
    def on_fill(self, event: Union[Event, FillEvent]):
        """Handle both legacy Event and semantic FillEvent."""
        if isinstance(event, FillEvent):
            # Direct access to typed fields!
            symbol = event.symbol
            quantity = event.quantity
            price = event.price
        else:
            # Legacy event - extract from payload
            symbol = event.payload.get('symbol')
            quantity = event.payload.get('quantity', 0)
            price = event.payload.get('price', 0)
        
        # Process the fill...
    
    def open_position(self, signal: TradingSignal):
        """Open position from semantic signal."""
        # Create semantic position event
        position_event = PositionOpenEvent(
            symbol=signal.symbol,
            quantity=self._calculate_size(signal),
            side="LONG" if signal.action == "BUY" else "SHORT",
            entry_price=self._get_current_price(signal.symbol),
            correlation_id=signal.correlation_id,
            causation_id=signal.event_id,
            source_container=self.container_id,
            strategy_id=signal.strategy_id
        )
        
        # Validate before publishing
        if position_event.validate():
            # Publish as semantic event
            self.event_bus.publish(position_event)
            
            # OR convert to legacy if needed
            # self.event_bus.publish(position_event.to_legacy_event())


# ============================================
# STEP 3: Update Your MetricsEventTracer
# ============================================
"""
Update the tracer to handle semantic events efficiently:
"""

# In src/core/events/observers/tracer.py, update trace_event method:
class EventTracer:
    def trace_event(self, event: Union[Event, SemanticEventProtocol]) -> None:
        """Trace both legacy and semantic events."""
        
        # Handle semantic events
        if hasattr(event, 'event_id') and hasattr(event, 'correlation_id'):
            # It's a semantic event
            event_dict = event.to_dict() if hasattr(event, 'to_dict') else asdict(event)
            
            # Add trace metadata
            event_dict['trace_id'] = self.trace_id
            event_dict['traced_at'] = datetime.now().isoformat()
            
            # Store the semantic event
            self.storage.store(event_dict)
            self._traced_count += 1
            
            # Apply retention policy for specific semantic events
            if isinstance(event, PositionCloseEvent):
                # Prune related position events
                self._apply_position_retention(event)
        else:
            # Legacy event handling (existing code)
            self._trace_legacy_event(event)
    
    def _apply_position_retention(self, close_event: PositionCloseEvent):
        """Apply retention policy for position events."""
        if self.retention_policy == 'trade_complete':
            # Remove all events for this position
            pruned = self.storage.prune({
                'correlation_id': close_event.correlation_id,
                '__type__': 'PositionOpenEvent'
            })
            self._pruned_count += pruned


# ============================================
# STEP 4: Update Your Routes to Validate Semantic Events
# ============================================
"""
Your routes can now validate semantic properties:
"""

# In src/core/routing/pipe.py or other routes:
class PipelineRoute:
    def _validate_event_routing(self, event: Any, source: Container, target: Container):
        """Enhanced validation for semantic events."""
        
        # Check if it's a semantic event
        if hasattr(event, 'validate'):
            is_valid, errors = validate_semantic_event(event)
            if not is_valid:
                self.logger.warning(f"Invalid semantic event: {errors}")
                if self.config.get('strict_validation', False):
                    raise ValueError(f"Invalid event: {errors}")
        
        # Type-specific validation
        if isinstance(event, TradingSignal):
            # Validate signal strength for target
            if hasattr(target, 'minimum_signal_strength'):
                if event.strength < target.minimum_signal_strength:
                    self.logger.warning(
                        f"Signal strength {event.strength} below "
                        f"target minimum {target.minimum_signal_strength}"
                    )


# ============================================
# STEP 5: Migration Strategy
# ============================================
"""
You can migrate gradually - both systems work together!
"""

# Helper to convert legacy events to semantic
def legacy_to_semantic(event: Event) -> Optional[SemanticEventProtocol]:
    """Convert legacy Event to semantic event if possible."""
    
    if event.event_type == EventType.FILL:
        payload = event.payload
        return FillEvent(
            order_id=payload.get('order_id', ''),
            symbol=payload.get('symbol', ''),
            side=payload.get('side', 'BUY'),
            quantity=payload.get('quantity', 0),
            price=payload.get('price', 0.0),
            correlation_id=event.correlation_id or make_correlation_id(),
            source_container=event.source_id or '',
            timestamp=event.timestamp
        )
    
    # Add more conversions as needed
    return None

# Helper to work with both types
def get_event_symbol(event: Union[Event, TradingSignal, FillEvent]) -> str:
    """Extract symbol from any event type."""
    if hasattr(event, 'symbol'):
        return event.symbol  # Semantic event
    elif hasattr(event, 'payload'):
        return event.payload.get('symbol', '')  # Legacy event
    return ''


# ============================================
# STEP 6: Update Your __init__.py Files
# ============================================
"""
Update imports to include semantic events:
"""

# In src/core/events/__init__.py, add:
from .semantic import (
    # Protocols
    SemanticEventProtocol,
    
    # Trading events
    PositionOpenEvent,
    PositionCloseEvent,
    TradingSignal,
    OrderEvent,
    FillEvent,
    
    # Utilities
    create_event_with_context,
    create_caused_event,
    validate_semantic_event,
    is_semantic_event,
)

# Add to __all__:
__all__ = [
    # ... existing exports ...
    
    # Semantic events
    'SemanticEventProtocol',
    'PositionOpenEvent',
    'PositionCloseEvent',
    'TradingSignal',
    'OrderEvent',
    'FillEvent',
    'create_event_with_context',
    'create_caused_event',
    'validate_semantic_event',
    'is_semantic_event',
]


# ============================================
# USAGE EXAMPLES
# ============================================
"""
Here's how to use semantic events in your system:
"""

# 1. In a strategy container
class MomentumStrategy:
    def process_features(self, features: Dict[str, float]):
        """Generate semantic trading signal."""
        
        signal_strength = self._calculate_signal(features)
        
        if abs(signal_strength) > self.threshold:
            signal = TradingSignal(
                symbol=self.symbol,
                action="BUY" if signal_strength > 0 else "SELL",
                strength=abs(signal_strength),
                confidence=self._calculate_confidence(features),
                source_container=self.container_id,
                source_component="momentum_strategy",
                strategy_id=self.strategy_id,
                indicators_used={
                    'rsi': features.get('rsi', 0),
                    'macd': features.get('macd', 0)
                }
            )
            
            # Validate and publish
            if signal.validate():
                self.event_bus.publish(signal)

# 2. In a portfolio container with minimal memory mode
class MinimalPortfolioContainer:
    def __init__(self):
        self.open_positions: Dict[str, PositionOpenEvent] = {}
    
    def on_position_open(self, event: PositionOpenEvent):
        """Track position using semantic event."""
        # Store the semantic event directly!
        self.open_positions[event.correlation_id] = event
    
    def on_position_close(self, event: PositionCloseEvent):
        """Close position and calculate metrics."""
        # Get the open event
        open_event = self.open_positions.pop(event.correlation_id, None)
        if open_event:
            # Everything is strongly typed!
            self.metrics.update_from_trade(
                entry_price=open_event.entry_price,
                exit_price=event.exit_price,
                quantity=event.quantity,
                direction='long' if open_event.side == 'LONG' else 'short'
            )

# 3. Event chaining with semantic events
def process_market_data(container, market_data: MarketDataEvent):
    """Process market data through the system."""
    
    # Generate indicator
    indicator_event = create_caused_event(
        market_data,
        IndicatorEvent,
        indicator_name="RSI",
        value=calculate_rsi(market_data),
        source_container=container.container_id
    )
    container.event_bus.publish(indicator_event)
    
    # Generate signal from indicator
    if indicator_event.value > 0.7:
        signal = create_caused_event(
            indicator_event,
            TradingSignal,
            symbol=market_data.symbol,
            action="SELL",
            strength=0.8,
            source_container=container.container_id
        )
        container.event_bus.publish(signal)
    
    # Complete traceability through correlation_id and causation_id!


# ============================================
# BENEFITS OF THIS APPROACH
# ============================================
"""
1. **Gradual Migration**: Use semantic events where they help, keep legacy where they work
2. **Type Safety**: IDE knows all fields, autocomplete works, type checking catches errors
3. **Domain Modeling**: Events model your trading domain, not generic data bags
4. **Traceability**: Every event knows its lineage through correlation/causation
5. **Performance**: Direct field access is faster than dict lookups
6. **Validation**: Events validate themselves
7. **No Breaking Changes**: Existing code continues to work
"""
