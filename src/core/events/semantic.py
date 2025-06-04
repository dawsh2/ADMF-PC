"""
Semantic events for ADMF-PC using protocol-based design.

This module provides strongly-typed, versioned, and traceable events
without inheritance, following ADMF-PC's protocol-based architecture.
"""

from typing import Protocol, runtime_checkable, Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import math


def make_event_id() -> str:
    """Generate unique event ID."""
    return str(uuid.uuid4())


def make_correlation_id() -> str:
    """Generate correlation ID for event flow."""
    return str(uuid.uuid4())


@runtime_checkable
class SemanticEvent(Protocol):
    """Protocol defining what makes something a semantic event.
    
    Following ADMF-PC's no-inheritance principle, any object with these
    fields IS a semantic event - no base class required.
    """
    
    # Required fields for all semantic events
    event_id: str
    schema_version: str
    timestamp: datetime
    
    # Correlation tracking for event lineage
    correlation_id: str
    causation_id: Optional[str]
    
    # Source information for debugging
    source_container: str
    source_component: str
    
    # Business context for trading
    strategy_id: Optional[str]
    portfolio_id: Optional[str]
    regime_context: Optional[str]
    
    def validate(self) -> bool:
        """Validate the event's business logic."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        ...


class EventCategory(Enum):
    """Categories of semantic events in trading system."""
    MARKET_DATA = "market_data"
    INDICATOR = "indicator"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    SYSTEM = "system"


# Trading-specific semantic events using dataclasses

@dataclass
class MarketDataEvent:
    """Market data event - no inheritance needed!"""
    
    # Event protocol fields
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "1.2.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None
    source_container: str = ""
    source_component: str = ""
    
    # Business context
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    regime_context: Optional[str] = None
    
    # Market data fields
    symbol: str = ""
    price: float = 0.0
    volume: int = 0
    bid: Optional[float] = None
    ask: Optional[float] = None
    data_type: Literal["BAR", "TICK", "QUOTE"] = "BAR"
    timeframe: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate market data."""
        return (
            self.price > 0 and 
            self.volume >= 0 and 
            bool(self.symbol.strip())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_obj in self.__dataclass_fields__.values():
            value = getattr(self, field_obj.name)
            if isinstance(value, datetime):
                result[field_obj.name] = value.isoformat()
            else:
                result[field_obj.name] = value
        return result


@dataclass
class FeatureEvent:
    """Technical feature event."""
    
    # Event protocol fields
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "1.1.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None
    source_container: str = ""
    source_component: str = ""
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    regime_context: Optional[str] = None
    
    # Feature fields
    symbol: str = ""
    feature_name: str = ""
    value: float = 0.0
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate feature."""
        return (
            bool(self.feature_name.strip()) and 
            not math.isnan(self.value) and
            bool(self.symbol.strip())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_obj in self.__dataclass_fields__.values():
            value = getattr(self, field_obj.name)
            if isinstance(value, datetime):
                result[field_obj.name] = value.isoformat()
            else:
                result[field_obj.name] = value
        return result


@dataclass
class TradingSignal:
    """Trading signal event."""
    
    # Event protocol fields
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "2.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None
    source_container: str = ""
    source_component: str = ""
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    regime_context: Optional[str] = None
    
    # Signal-specific fields
    symbol: str = ""
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    strength: float = 0.0  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_score: float = 0.5  # 0.0 to 1.0
    
    def validate(self) -> bool:
        """Validate trading signal."""
        return (
            0.0 <= self.strength <= 1.0 and
            bool(self.symbol.strip()) and
            0.0 <= self.risk_score <= 1.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_obj in self.__dataclass_fields__.values():
            value = getattr(self, field_obj.name)
            if isinstance(value, datetime):
                result[field_obj.name] = value.isoformat()
            else:
                result[field_obj.name] = value
        return result


@dataclass
class OrderEvent:
    """Order placement event."""
    
    # Event protocol fields
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None
    source_container: str = ""
    source_component: str = ""
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    regime_context: Optional[str] = None
    
    # Order fields
    order_type: Literal["MARKET", "LIMIT", "STOP"] = "MARKET"
    symbol: str = ""
    quantity: int = 0
    price: Optional[float] = None
    side: Literal["BUY", "SELL"] = "BUY"
    max_position_pct: float = 0.02
    stop_loss_pct: Optional[float] = None
    
    def validate(self) -> bool:
        """Validate order."""
        return (
            self.quantity > 0 and
            bool(self.symbol.strip()) and
            0.0 < self.max_position_pct <= 1.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_obj in self.__dataclass_fields__.values():
            value = getattr(self, field_obj.name)
            if isinstance(value, datetime):
                result[field_obj.name] = value.isoformat()
            else:
                result[field_obj.name] = value
        return result


@dataclass
class FillEvent:
    """Order fill event."""
    
    # Event protocol fields
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None  # Usually the OrderEvent.event_id
    source_container: str = ""
    source_component: str = ""
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    regime_context: Optional[str] = None
    
    # Fill fields
    symbol: str = ""
    quantity: int = 0
    fill_price: float = 0.0
    side: Literal["BUY", "SELL"] = "BUY"
    commission: float = 0.0
    order_id: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate fill."""
        return (
            self.quantity > 0 and
            self.fill_price > 0.0 and
            bool(self.symbol.strip()) and
            self.commission >= 0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_obj in self.__dataclass_fields__.values():
            value = getattr(self, field_obj.name)
            if isinstance(value, datetime):
                result[field_obj.name] = value.isoformat()
            else:
                result[field_obj.name] = value
        return result


@dataclass
class PortfolioUpdateEvent:
    """Portfolio state update event."""
    
    # Event protocol fields
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None  # Usually the FillEvent.event_id
    source_container: str = ""
    source_component: str = ""
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    regime_context: Optional[str] = None
    
    # Portfolio fields
    total_value: float = 0.0
    cash: float = 0.0
    positions: Dict[str, int] = field(default_factory=dict)  # symbol -> quantity
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def validate(self) -> bool:
        """Validate portfolio update."""
        return (
            self.total_value >= 0.0 and
            self.cash >= 0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_obj in self.__dataclass_fields__.values():
            value = getattr(self, field_obj.name)
            if isinstance(value, datetime):
                result[field_obj.name] = value.isoformat()
            else:
                result[field_obj.name] = value
        return result


# Event creation helpers

def create_event_with_context(event_class, **kwargs):
    """Create event with automatic context from current container."""
    # This would integrate with container context in real implementation
    return event_class(**kwargs)


def create_caused_event(causing_event: SemanticEvent, event_class, **kwargs):
    """Create event caused by another event, preserving lineage."""
    return event_class(
        correlation_id=causing_event.correlation_id,
        causation_id=causing_event.event_id,
        source_container=kwargs.get('source_container', causing_event.source_container),
        **kwargs
    )


def validate_semantic_event(event: Any) -> bool:
    """Validate any object claiming to be a semantic event."""
    
    # Check if it implements the protocol
    if not isinstance(event, SemanticEvent):
        return False
    
    # Check required fields exist
    required_fields = [
        'event_id', 'schema_version', 'timestamp',
        'correlation_id', 'source_container'
    ]
    
    for field in required_fields:
        if not hasattr(event, field):
            return False
    
    # Call event's own validation
    if hasattr(event, 'validate'):
        return event.validate()
    
    return True


# Type transformations for type flow analysis

def feature_to_signal(feature: FeatureEvent, **signal_kwargs) -> TradingSignal:
    """Transform feature event to trading signal."""
    
    # Determine action from feature value
    if feature.value > 0.6:
        action = "BUY"
        strength = min(feature.value, 1.0)
    elif feature.value < -0.6:
        action = "SELL"
        strength = min(abs(feature.value), 1.0)
    else:
        action = "HOLD"
        strength = 0.0
    
    return TradingSignal(
        # Preserve lineage
        correlation_id=feature.correlation_id,
        causation_id=feature.event_id,
        source_container=feature.source_container,
        
        # Transform data
        symbol=feature.symbol,
        action=action,
        strength=strength,
        strategy_id=feature.strategy_id,
        regime_context=feature.regime_context,
        
        # Override with any provided kwargs
        **signal_kwargs
    )


def signal_to_order(signal: TradingSignal, quantity: int, **order_kwargs) -> OrderEvent:
    """Transform trading signal to order."""
    return OrderEvent(
        # Preserve lineage
        correlation_id=signal.correlation_id,
        causation_id=signal.event_id,
        source_container=signal.source_container,
        
        # Create order
        symbol=signal.symbol,
        side="BUY" if signal.action == "BUY" else "SELL",
        order_type="MARKET",
        quantity=quantity,
        strategy_id=signal.strategy_id,
        
        # Override with any provided kwargs
        **order_kwargs
    )


def order_to_fill(order: OrderEvent, fill_price: float, **fill_kwargs) -> FillEvent:
    """Transform order to fill event."""
    return FillEvent(
        # Preserve lineage
        correlation_id=order.correlation_id,
        causation_id=order.event_id,
        source_container=order.source_container,
        
        # Create fill
        symbol=order.symbol,
        quantity=order.quantity,
        side=order.side,
        fill_price=fill_price,
        order_id=order.event_id,
        strategy_id=order.strategy_id,
        
        # Override with any provided kwargs
        **fill_kwargs
    )