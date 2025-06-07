# Routing Refactor Proposal

## Problems with Current Design

### 1. RiskServiceRoute mixes concerns:
- **Routing logic**: Subscribe to portfolios, publish to root bus
- **Business logic**: Risk validation
- **Event transformation**: ORDER_REQUEST → ORDER

### 2. FeatureDispatcher mixes concerns:
- **Routing logic**: Subscribe to features, publish to strategies  
- **Business logic**: Feature filtering based on requirements
- **Registry management**: Tracking strategy requirements

## Proposed Solution: Separation of Concerns

### 1. Generic ProcessingRoute Pattern

```python
class ProcessingRoute:
    """
    Generic route that applies processing to events.
    
    Separates routing (where events go) from processing (what happens to them).
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.source_pattern = config.get('source_pattern')  # e.g., 'portfolio_*'
        self.target = config.get('target')  # e.g., root_event_bus
        self.processor = config.get('processor')  # The business logic component
        self.input_event_types = config.get('input_types', [])
        self.output_event_type = config.get('output_type')
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Route events through processor."""
        # Let processor handle the business logic
        result = self.processor.process(event, source)
        
        if result:
            # Publish transformed event to target
            self.target.publish(result)
```

### 2. Refactor RiskServiceRoute

```python
# Separate the risk validation logic
class RiskValidationProcessor:
    """Pure business logic for risk validation."""
    
    def __init__(self, risk_validators: Dict[str, Any]):
        self.risk_validators = risk_validators
        
    def process(self, event: Event, source: Container) -> Optional[Event]:
        """Validate ORDER_REQUEST and return ORDER or None."""
        if event.event_type != EventType.ORDER_REQUEST:
            return None
            
        order = event.payload.get('order')
        risk_params = event.payload.get('risk_params')
        
        # Apply validation logic
        validator = self._get_validator(risk_params)
        if validator and validator.validate(order):
            # Transform to ORDER event
            return Event(
                event_type=EventType.ORDER,
                payload=order,
                metadata=event.metadata
            )
        return None

# Use generic ProcessingRoute with RiskValidationProcessor
risk_route = ProcessingRoute(
    name='risk_validation',
    config={
        'source_pattern': 'portfolio_*',
        'target': root_event_bus,
        'processor': RiskValidationProcessor(risk_validators),
        'input_types': [EventType.ORDER_REQUEST],
        'output_type': EventType.ORDER
    }
)
```

### 3. Refactor FeatureDispatcher

```python
# Separate feature filtering logic
class FeatureFilter:
    """Pure feature filtering logic."""
    
    def __init__(self):
        self.strategy_requirements = {}
        
    def register_requirements(self, strategy_id: str, requirements: Set[str]):
        self.strategy_requirements[strategy_id] = requirements
        
    def filter_for_strategy(self, features: Dict, strategy_id: str) -> Dict:
        """Filter features based on strategy requirements."""
        requirements = self.strategy_requirements.get(strategy_id, set())
        return {k: v for k, v in features.items() if k in requirements}

# Use with FanOutRoute
feature_route = FanOutRoute(
    name='feature_dispatcher',
    config={
        'source': 'feature_container',
        'targets': [
            {
                'name': strategy_id,
                'transform': lambda e: transform_features_for_strategy(e, strategy_id)
            }
            for strategy_id in strategies
        ]
    }
)
```

## Benefits

1. **Single Responsibility**: Routes handle routing, processors handle logic
2. **Reusability**: ProcessingRoute can be used for any transform/validate pattern
3. **Testability**: Business logic separated from routing infrastructure
4. **Composability**: Can chain processors or swap them out
5. **Clarity**: Clear separation between "where" and "what"

## Generic Route Patterns

### 1. CollectorRoute
- Collects events from multiple sources matching a pattern
- Example: All portfolio containers

### 2. ProcessingRoute  
- Applies processing/transformation to events
- Example: Risk validation, feature filtering

### 3. FanOutRoute
- One source to many targets with optional transforms
- Example: Features to strategies

### 4. SelectiveRoute
- Routes based on event content
- Example: Signals to specific portfolios

### 5. BroadcastRoute
- Simple one-to-many without transformation
- Example: Fills to all portfolios