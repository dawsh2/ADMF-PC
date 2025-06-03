# Semantic Events

## Overview

The ADMF-PC semantic event system provides strongly-typed, versioned, and traceable events that form the communication backbone of the trading system. Unlike generic event objects with arbitrary data payloads, semantic events are domain-specific, type-safe, and carry complete lineage information for debugging and compliance.

## Core Philosophy: Protocol-Based Events

Following ADMF-PC's zero-inheritance principle, semantic events are defined through protocols and composition, not base classes. Any data structure that has the right fields can be a semantic event - it doesn't need to inherit from anything.

### What Makes Events "Semantic"?

Semantic events differ from generic events in several key ways:

1. **Domain Modeling**: Events model specific trading concepts (signals, orders, fills) rather than generic data containers
2. **Type Safety**: Full IDE support with autocomplete, type checking, and compile-time validation
3. **Lineage Tracking**: Every event knows what caused it and what it's correlated with
4. **Schema Evolution**: Events can evolve over time without breaking existing systems
5. **Self-Describing**: Events carry their schema version and validation rules

### The Problem with Generic Events

Traditional event systems suffer from several issues:

```python
# Traditional approach - no type safety, no validation
event = Event(
    type="SIGNAL",
    data={"symbol": "AAPL", "action": "BUY", "strength": 0.8}
)

# Problems:
# - No IDE autocomplete for event.data fields
# - Runtime errors if fields are missing or wrong type
# - No way to track what caused this event
# - Schema changes break consumers silently
# - Difficult to debug event flows in production
```

### The Semantic Event Solution

```python
# Semantic approach - full type safety and traceability
signal = TradingSignal(
    symbol="AAPL",
    action="BUY",
    strength=0.85,
    regime_context="BULL",
    causation_id=indicator_event.event_id,
    correlation_id=flow.correlation_id
)

# Benefits:
# - IDE knows all fields and types
# - Validation happens automatically
# - Complete audit trail of event causation
# - Schema versioning prevents breaking changes
# - Easy to trace through complex workflows
```

## Event Protocol Definition

Instead of inheritance, we define what makes something a semantic event through protocols:

```python
from typing import Protocol, runtime_checkable, Optional, Dict, Any
from datetime import datetime

@runtime_checkable
class SemanticEvent(Protocol):
    """Protocol defining what makes something a semantic event"""
    
    # Required fields
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
    regime_context: Optional[str]
    
    def validate(self) -> bool:
        """Validate the event"""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        ...
```

Any object that implements this protocol IS a semantic event. No inheritance required!

## Creating Semantic Events with Dataclasses

The easiest way to create semantic events is with dataclasses:

```python
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
from datetime import datetime
import uuid

def make_event_id() -> str:
    """Generate unique event ID"""
    return str(uuid.uuid4())

def make_correlation_id() -> str:
    """Generate correlation ID for event flow"""
    return str(uuid.uuid4())

@dataclass
class TradingSignal:
    """Trading signal event - no inheritance needed!"""
    
    # Event protocol fields
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "2.0.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None
    source_container: str = ""
    source_component: str = ""
    
    # Business context
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    regime_context: Optional[str] = None
    
    # Signal-specific fields
    symbol: str = ""
    action: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    strength: float = 0.0  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_score: float = 0.5
    
    def validate(self) -> bool:
        """Validate signal"""
        return (
            0.0 <= self.strength <= 1.0 and
            self.symbol and
            0.0 <= self.risk_score <= 1.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
```

## Event Type Library

### Market Data Events

```python
@dataclass
class MarketDataEvent:
    """Market data event"""
    # Event protocol fields
    event_id: str = field(default_factory=make_event_id)
    schema_version: str = "1.2.0"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str = field(default_factory=make_correlation_id)
    causation_id: Optional[str] = None
    source_container: str = ""
    source_component: str = ""
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
        return self.price > 0 and self.volume >= 0
```

### Indicator Events

```python
@dataclass
class IndicatorEvent:
    """Technical indicator event"""
    # Event protocol fields (same pattern)
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
    
    # Indicator fields
    indicator_name: str = ""
    value: float = 0.0
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        return self.indicator_name and not math.isnan(self.value)
```

### Order Events

```python
@dataclass
class OrderEvent:
    """Order placement event"""
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
        return (
            self.quantity > 0 and
            self.symbol and
            0.0 < self.max_position_pct <= 1.0
        )
```

## Event Helpers and Utilities

### Event Creation Helpers

```python
def create_event_with_context(event_class, **kwargs):
    """Create event with automatic context"""
    # Get context from current container
    context = get_current_container_context()
    
    return event_class(
        source_container=context.container_name,
        source_component=context.component_name,
        strategy_id=context.strategy_id,
        portfolio_id=context.portfolio_id,
        **kwargs
    )

# Usage
signal = create_event_with_context(
    TradingSignal,
    symbol="AAPL",
    action="BUY",
    strength=0.8
)
```

### Event Validation

```python
def validate_semantic_event(event: Any) -> bool:
    """Validate any object claiming to be a semantic event"""
    
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
```

## Schema Evolution

### Version Management

```python
# Simple version registry - no inheritance!
EVENT_SCHEMAS = {
    'TradingSignal': {
        '1.0.0': {'fields': ['symbol', 'action', 'strength']},
        '2.0.0': {'fields': ['symbol', 'action', 'strength', 'risk_score', 'regime_context']}
    },
    'OrderEvent': {
        '1.0.0': {'fields': ['symbol', 'quantity', 'side', 'order_type']}
    }
}

def get_schema_version(event_type: str, version: str) -> Dict[str, Any]:
    """Get schema for specific version"""
    return EVENT_SCHEMAS.get(event_type, {}).get(version, {})
```

### Migration Functions

```python
# Migration functions - pure functions, no classes
def migrate_trading_signal_v1_to_v2(v1_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate TradingSignal from v1.0.0 to v2.0.0"""
    v2_signal = v1_signal.copy()
    v2_signal['schema_version'] = '2.0.0'
    v2_signal['risk_score'] = 0.5  # Default value
    v2_signal['regime_context'] = None  # Optional field
    return v2_signal

# Registry of migrations
MIGRATIONS = {
    ('TradingSignal', '1.0.0', '2.0.0'): migrate_trading_signal_v1_to_v2,
}

def migrate_event(event_dict: Dict[str, Any], target_version: str) -> Dict[str, Any]:
    """Migrate event to target version"""
    event_type = event_dict.get('__type__', type(event_dict).__name__)
    current_version = event_dict.get('schema_version', '1.0.0')
    
    if current_version == target_version:
        return event_dict
    
    migration_key = (event_type, current_version, target_version)
    if migration_key in MIGRATIONS:
        return MIGRATIONS[migration_key](event_dict)
    
    raise ValueError(f"No migration path from {current_version} to {target_version}")
```

## Event Transformation

### Type-Safe Transformations

```python
def indicator_to_signal(indicator: IndicatorEvent) -> TradingSignal:
    """Transform indicator event to trading signal"""
    return TradingSignal(
        # Preserve lineage
        correlation_id=indicator.correlation_id,
        causation_id=indicator.event_id,
        source_container=indicator.source_container,
        
        # Transform data
        symbol=indicator.metadata.get('symbol', ''),
        action="BUY" if indicator.value > 0 else "SELL",
        strength=abs(indicator.value),
        strategy_id=indicator.strategy_id,
        regime_context=indicator.regime_context
    )

def signal_to_order(signal: TradingSignal, position_sizer) -> OrderEvent:
    """Transform trading signal to order"""
    quantity = position_sizer.calculate_size(signal)
    
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
        strategy_id=signal.strategy_id
    )
```

## Event Serialization

### JSON Serialization

```python
import json
from datetime import datetime

def serialize_event(event: Any) -> str:
    """Serialize semantic event to JSON"""
    event_dict = event.to_dict() if hasattr(event, 'to_dict') else vars(event)
    
    # Handle special types
    for key, value in event_dict.items():
        if isinstance(value, datetime):
            event_dict[key] = value.isoformat()
    
    # Add type information
    event_dict['__type__'] = type(event).__name__
    
    return json.dumps(event_dict)

def deserialize_event(json_str: str) -> Any:
    """Deserialize JSON to semantic event"""
    data = json.loads(json_str)
    
    # Get event type
    event_type = data.pop('__type__', 'GenericEvent')
    
    # Handle datetime fields
    if 'timestamp' in data:
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
    
    # Create appropriate event type
    if event_type == 'TradingSignal':
        return TradingSignal(**data)
    elif event_type == 'OrderEvent':
        return OrderEvent(**data)
    # ... etc
```

## Event Flow Tracking

### Correlation and Causation

```python
def create_correlated_flow() -> str:
    """Create a new correlation ID for a flow"""
    return make_correlation_id()

def create_caused_event(causing_event: Any, event_class, **kwargs):
    """Create event caused by another event"""
    return event_class(
        correlation_id=causing_event.correlation_id,
        causation_id=causing_event.event_id,
        **kwargs
    )

# Example flow
flow_id = create_correlated_flow()

# Market data starts the flow
market_data = MarketDataEvent(
    correlation_id=flow_id,
    symbol="AAPL",
    price=150.0
)

# Indicator caused by market data
indicator = create_caused_event(
    market_data,
    IndicatorEvent,
    indicator_name="RSI",
    value=0.7
)

# Signal caused by indicator
signal = create_caused_event(
    indicator,
    TradingSignal,
    symbol="AAPL",
    action="BUY",
    strength=0.8
)

# Complete traceability!
```

## Integration with Adapters

### Semantic Event Filtering

```python
# Adapters can filter based on semantic properties
def high_confidence_filter(event: Any) -> bool:
    """Filter for high confidence events"""
    if hasattr(event, 'confidence') and event.confidence > 0.8:
        return True
    if hasattr(event, 'strength') and event.strength > 0.7:
        return True
    return False

def regime_filter(target_regime: str):
    """Create filter for specific regime"""
    def filter_fn(event: Any) -> bool:
        return hasattr(event, 'regime_context') and event.regime_context == target_regime
    return filter_fn

# Use in adapter configuration
adapter_config = {
    'type': 'selective_broadcast',
    'source': 'signal_generator',
    'targets': [
        {
            'name': 'aggressive_strategy',
            'filter': high_confidence_filter
        },
        {
            'name': 'bull_market_strategy',
            'filter': regime_filter('BULL')
        }
    ]
}
```

## Best Practices

### 1. Event Design

- **Keep Events Focused**: Each event type should represent one concept
- **Include Context**: Always populate correlation and causation IDs
- **Version from Start**: Include schema_version in initial design
- **Validate Early**: Implement validate() method for each event type

### 2. Event Usage

```python
# Good: Create events with full context
signal = TradingSignal(
    symbol="AAPL",
    action="BUY",
    strength=0.8,
    causation_id=previous_event.event_id,
    correlation_id=flow.correlation_id,
    source_container=self.container_name
)

# Bad: Missing context
signal = TradingSignal(symbol="AAPL", action="BUY")  # No lineage!
```

### 3. Schema Evolution

- **Add, Don't Remove**: New fields should be optional
- **Provide Defaults**: Migrations should provide sensible defaults
- **Document Changes**: Keep clear documentation of version changes
- **Test Migrations**: Ensure events can round-trip through versions

## Testing Semantic Events

```python
def test_event_protocol_compliance(event_class):
    """Test that event class implements protocol correctly"""
    # Create instance
    event = event_class()
    
    # Check protocol compliance
    assert isinstance(event, SemanticEvent)
    
    # Check required methods
    assert hasattr(event, 'validate')
    assert hasattr(event, 'to_dict')
    
    # Check validation works
    assert isinstance(event.validate(), bool)

def test_event_lineage():
    """Test event causation chain"""
    # Create flow
    flow_id = create_correlated_flow()
    
    # Create chain
    event1 = MarketDataEvent(correlation_id=flow_id)
    event2 = create_caused_event(event1, IndicatorEvent)
    event3 = create_caused_event(event2, TradingSignal)
    
    # Verify chain
    assert event2.causation_id == event1.event_id
    assert event3.causation_id == event2.event_id
    assert all(e.correlation_id == flow_id for e in [event1, event2, event3])
```

## Type Flow Integration

Semantic events integrate seamlessly with the Type Flow Analysis system to provide compile-time and runtime guarantees about event routing.

### Event Type Registry

Register semantic events with the type flow system:

```python
from enum import Enum
from typing import Type, Dict, Set

class EventTypeRegistry:
    """Registry for semantic event types and their relationships"""
    
    def __init__(self):
        # Map event classes to EventType enum values
        self.event_types: Dict[Type, EventType] = {
            MarketDataEvent: EventType.BAR,
            IndicatorEvent: EventType.INDICATOR,
            TradingSignal: EventType.SIGNAL,
            OrderEvent: EventType.ORDER,
            FillEvent: EventType.FILL,
            PortfolioUpdateEvent: EventType.PORTFOLIO_UPDATE,
        }
        
        # Define which events can transform to which
        self.transformations: Dict[EventType, Set[EventType]] = {
            EventType.BAR: {EventType.INDICATOR, EventType.SIGNAL},
            EventType.INDICATOR: {EventType.SIGNAL},
            EventType.SIGNAL: {EventType.ORDER},
            EventType.ORDER: {EventType.FILL},
            EventType.FILL: {EventType.PORTFOLIO_UPDATE},
        }
    
    def register_event_type(self, event_class: Type, event_type: EventType):
        """Register a new semantic event type"""
        self.event_types[event_class] = event_type
    
    def get_event_type(self, event: Any) -> Optional[EventType]:
        """Get the EventType for a semantic event instance"""
        return self.event_types.get(type(event))
    
    def can_transform(self, from_type: EventType, to_type: EventType) -> bool:
        """Check if one event type can transform to another"""
        return to_type in self.transformations.get(from_type, set())
```

### Container Type Inference

Automatically infer container types from the semantic events they handle:

```python
class ContainerTypeInferencer:
    """Infer container types from semantic events"""
    
    def __init__(self, registry: EventTypeRegistry):
        self.registry = registry
        
    def infer_container_type(self, container: Container) -> str:
        """Infer container type from its event handlers"""
        
        # Check what events the container produces
        if hasattr(container, 'produces_events'):
            produced = container.produces_events()
            
            if MarketDataEvent in produced:
                return 'data_source'
            elif IndicatorEvent in produced:
                return 'indicator_engine'
            elif TradingSignal in produced:
                return 'strategy'
            elif OrderEvent in produced:
                return 'risk_manager'
            elif FillEvent in produced:
                return 'execution_engine'
            elif PortfolioUpdateEvent in produced:
                return 'portfolio_manager'
        
        # Fallback to name-based inference
        return self._infer_from_name(container.name)
    
    def get_expected_inputs(self, container: Container) -> Set[Type]:
        """Get expected input event types for a container"""
        container_type = self.infer_container_type(container)
        
        # Map container types to expected inputs
        expected_inputs = {
            'indicator_engine': {MarketDataEvent},
            'strategy': {MarketDataEvent, IndicatorEvent},
            'risk_manager': {TradingSignal},
            'execution_engine': {OrderEvent},
            'portfolio_manager': {FillEvent},
        }
        
        return expected_inputs.get(container_type, set())
```

### Semantic Event Flow Validation

Validate that semantic events flow correctly through adapters:

```python
class SemanticEventFlowValidator:
    """Validate semantic event flows"""
    
    def __init__(self, registry: EventTypeRegistry, 
                 type_analyzer: TypeFlowAnalyzer):
        self.registry = registry
        self.type_analyzer = type_analyzer
        
    def validate_event_flow(self, event: SemanticEvent, 
                          source: Container, 
                          target: Container) -> ValidationResult:
        """Validate a specific event flow"""
        
        # Get event type
        event_type = self.registry.get_event_type(event)
        if not event_type:
            return ValidationResult(
                valid=False,
                error=f"Unknown event type: {type(event).__name__}"
            )
        
        # Check if target can handle this event type
        target_inputs = self.type_analyzer.get_container_inputs(target.name)
        if event_type not in target_inputs:
            return ValidationResult(
                valid=False,
                error=f"{target.name} cannot handle {event_type} events"
            )
        
        # Validate semantic properties
        if not event.validate():
            return ValidationResult(
                valid=False,
                error=f"Event validation failed: {event}"
            )
        
        # Check schema compatibility
        if hasattr(target, 'required_schema_version'):
            required = target.required_schema_version(event_type)
            if event.schema_version != required:
                return ValidationResult(
                    valid=False,
                    error=f"Schema mismatch: {event.schema_version} != {required}",
                    can_migrate=self._can_migrate(event, required)
                )
        
        return ValidationResult(valid=True)
    
    def trace_event_lineage(self, event: SemanticEvent) -> List[SemanticEvent]:
        """Trace the complete lineage of an event"""
        lineage = [event]
        
        # Follow causation chain backwards
        current = event
        while current.causation_id:
            # Find parent event (would need event store)
            parent = self._find_event_by_id(current.causation_id)
            if parent:
                lineage.insert(0, parent)
                current = parent
            else:
                break
        
        return lineage
```

### Type Flow Visualization with Semantic Events

Enhanced visualization showing semantic event types:

```python
class SemanticTypeFlowVisualizer(TypeFlowVisualizer):
    """Visualize type flow with semantic event information"""
    
    def __init__(self, analyzer: TypeFlowAnalyzer, 
                 registry: EventTypeRegistry):
        super().__init__(analyzer)
        self.registry = registry
    
    def generate_semantic_flow_diagram(self, 
                                     flow_map: Dict[str, FlowNode]) -> str:
        """Generate diagram with semantic event details"""
        lines = ["graph TD", "    %% Semantic Event Flow"]
        
        # Add nodes with semantic event types
        for name, node in flow_map.items():
            # Get semantic event classes for this container
            semantic_types = self._get_semantic_types(node)
            
            if semantic_types:
                # Show actual event class names
                type_names = [t.__name__ for t in semantic_types]
                label = f"{name}<br/>[{', '.join(type_names)}]"
            else:
                label = name
                
            lines.append(f"    {name}[{label}]")
        
        # Add connections with transformation info
        connections = self.analyzer._build_connections(adapters)
        for source, targets in connections.items():
            source_node = flow_map.get(source)
            if not source_node:
                continue
                
            for target in targets:
                # Show what semantic transformations occur
                transformations = self._get_transformations(source, target)
                if transformations:
                    for from_event, to_event in transformations:
                        lines.append(
                            f"    {source} -->|{from_event.__name__} → "
                            f"{to_event.__name__}| {target}"
                        )
                else:
                    lines.append(f"    {source} --> {target}")
        
        return "\n".join(lines)
    
    def _get_semantic_types(self, node: FlowNode) -> Set[Type]:
        """Get semantic event types for a flow node"""
        semantic_types = set()
        
        # Map EventType enum to semantic classes
        for event_type in node.can_produce | node.can_receive:
            for event_class, mapped_type in self.registry.event_types.items():
                if mapped_type == event_type:
                    semantic_types.add(event_class)
        
        return semantic_types
```

### Usage Example

Integrating semantic events with type flow analysis:

```python
# Setup
registry = EventTypeRegistry()
type_analyzer = TypeFlowAnalyzer()
semantic_validator = SemanticEventFlowValidator(registry, type_analyzer)
visualizer = SemanticTypeFlowVisualizer(type_analyzer, registry)

# Register custom semantic events
@dataclass
class CustomAlertEvent:
    # ... semantic event fields ...
    pass

registry.register_event_type(CustomAlertEvent, EventType.SIGNAL)

# Validate event flow
market_data = MarketDataEvent(symbol="AAPL", price=150.0)
validation = semantic_validator.validate_event_flow(
    market_data, 
    data_source_container,
    indicator_container
)

if not validation.valid:
    print(f"Invalid flow: {validation.error}")
    if validation.can_migrate:
        # Perform automatic migration
        migrated = migrate_event(market_data, validation.required_version)

# Trace event lineage
signal = TradingSignal(
    symbol="AAPL",
    action="BUY",
    causation_id=indicator_event.event_id
)
lineage = semantic_validator.trace_event_lineage(signal)
print(f"Event lineage: {' → '.join(e.__class__.__name__ for e in lineage)}")

# Visualize semantic flow
flow_map = type_analyzer.analyze_flow(containers, adapters)
diagram = visualizer.generate_semantic_flow_diagram(flow_map)
print(diagram)
```

This integration provides:
- **Type-safe event routing**: Compile-time validation of event compatibility
- **Automatic inference**: Container types inferred from semantic events
- **Rich visualization**: See actual event classes in flow diagrams
- **Migration support**: Automatic schema migration when needed
- **Complete traceability**: Follow event lineage through the system

## Conclusion

Semantic events provide type safety, traceability, and evolution capabilities without requiring any inheritance. By using protocols and composition, any object with the right fields can participate in the semantic event system. This approach maintains maximum flexibility while providing all the benefits of a strongly-typed event system.

Key principles:
1. **No Inheritance**: Events are defined by protocols, not base classes
2. **Type Safety**: Full IDE support through dataclasses and type hints
3. **Traceability**: Every event knows its lineage through correlation and causation
4. **Evolution**: Schema versioning allows gradual system updates
5. **Simplicity**: Events are just data with validation - no complex hierarchies