# Routing Implementation Plan

## Phase 1: Create Generic ProcessingRoute

### 1. Create ProcessingRoute class

```python
# src/core/routing/core/processing.py
from typing import Dict, Any, Optional, Protocol, List
from ..protocols import Container, CommunicationRoute
from ...events import Event

class EventProcessor(Protocol):
    """Protocol for event processors."""
    def process(self, event: Event, source: Container) -> Optional[Event]:
        """Process event and return transformed event or None."""
        ...

class ProcessingRoute:
    """
    Generic route that processes events through a processor.
    
    This separates routing concerns from business logic.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.processor = config['processor']  # Required
        self.source_pattern = config.get('source_pattern')  # e.g., 'portfolio_*'
        self.source_names = config.get('sources', [])  # Explicit list
        self.target = config.get('target')  # Where to send processed events
        self.input_types = config.get('input_types', [])
        self.sources: List[Container] = []
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Setup connections based on pattern or explicit sources."""
        if self.source_pattern:
            # Find containers matching pattern
            import fnmatch
            self.sources = [
                container for name, container in containers.items()
                if fnmatch.fnmatch(name, self.source_pattern)
            ]
        else:
            # Use explicit source names
            self.sources = [
                containers[name] for name in self.source_names
                if name in containers
            ]
            
    def start(self) -> None:
        """Subscribe to source events."""
        for source in self.sources:
            for event_type in self.input_types:
                source.event_bus.subscribe(
                    event_type,
                    lambda e, s=source: self.handle_event(e, s)
                )
                
    def handle_event(self, event: Event, source: Container) -> None:
        """Process event through processor."""
        result = self.processor.process(event, source)
        if result and self.target:
            self.target.publish(result)
```

### 2. Refactor RiskServiceRoute as processor

```python
# src/core/routing/specialized/risk_processor.py
class RiskValidationProcessor:
    """
    Processes ORDER_REQUEST events through risk validation.
    
    This is pure business logic, no routing concerns.
    """
    
    def __init__(self, risk_validators: Dict[str, Any]):
        self.risk_validators = risk_validators
        self.metrics = {
            'processed': 0,
            'approved': 0,
            'rejected': 0
        }
        
    def process(self, event: Event, source: Container) -> Optional[Event]:
        """Validate ORDER_REQUEST and return ORDER event if approved."""
        if event.event_type != EventType.ORDER_REQUEST:
            return None
            
        self.metrics['processed'] += 1
        
        # Extract order and risk params
        order = event.payload.get('order')
        risk_params = event.payload.get('risk_params', {})
        portfolio_state = event.payload.get('portfolio_state')
        
        # Get appropriate validator
        risk_type = risk_params.get('type', 'default')
        validator = self.risk_validators.get(risk_type)
        
        if not validator:
            validator = self.risk_validators.get('default')
            
        # Validate
        if validator:
            result = validator.validate_order(
                order, portfolio_state, risk_params
            )
            
            if result.is_valid:
                self.metrics['approved'] += 1
                # Create ORDER event
                return Event(
                    event_type=EventType.ORDER,
                    payload=order,
                    metadata={
                        **event.metadata,
                        'risk_validated': True,
                        'risk_type': risk_type
                    }
                )
            else:
                self.metrics['rejected'] += 1
                # Could emit REJECTION event here
                
        return None
```

### 3. Update topology routing to use ProcessingRoute

```python
# In route_backtest_topology
if risk_validators:
    # Create risk processor
    risk_processor = RiskValidationProcessor(risk_validators)
    
    # Create processing route
    risk_route = routing_factory.create_route(
        name='risk_validation',
        config={
            'type': 'processing',  # New route type
            'processor': risk_processor,
            'source_pattern': 'portfolio_*',
            'target': root_event_bus,
            'input_types': [EventType.ORDER_REQUEST]
        }
    )
    routes.append(risk_route)
```

## Phase 2: Refactor FeatureDispatcher

### 1. Create FeatureFilterProcessor

```python
# src/core/routing/specialized/feature_processor.py
class FeatureFilterProcessor:
    """Filters features based on strategy requirements."""
    
    def __init__(self, strategy_requirements: Dict[str, Set[str]]):
        self.strategy_requirements = strategy_requirements
        
    def create_strategy_filter(self, strategy_id: str):
        """Create a filter function for a specific strategy."""
        requirements = self.strategy_requirements.get(strategy_id, set())
        
        def filter_features(event: Event) -> Event:
            """Filter features for this strategy."""
            if event.event_type != EventType.FEATURES:
                return event
                
            all_features = event.payload.get('features', {})
            filtered = {
                k: v for k, v in all_features.items() 
                if k in requirements
            }
            
            # Create new event with filtered features
            return Event(
                event_type=EventType.FEATURES,
                payload={
                    **event.payload,
                    'features': filtered,
                    'strategy_id': strategy_id
                },
                metadata=event.metadata
            )
            
        return filter_features
```

### 2. Use FanOutRoute with filters

```python
# In topology routing
if feature_containers and strategies:
    # Build strategy requirements
    feature_processor = FeatureFilterProcessor(strategy_requirements)
    
    # Create fan-out configuration
    targets = []
    for strategy_id, requirements in strategy_requirements.items():
        targets.append({
            'name': strategy_id,  # This would need to map to a container
            'transform': feature_processor.create_strategy_filter(strategy_id)
        })
    
    # Create fan-out route
    feature_route = routing_factory.create_route(
        name='feature_distribution',
        config={
            'type': 'fan_out',
            'source': 'features',  # Or iterate over feature containers
            'targets': targets
        }
    )
    routes.append(feature_route)
```

## Phase 3: Simplify Route Types

### Core Route Types (Keep)
1. **BroadcastRoute** - Simple one-to-many
2. **SelectiveRoute** - Content-based routing
3. **ProcessingRoute** - Transform with processor (NEW)
4. **FanOutRoute** - One-to-many with transforms

### Remove/Consolidate
1. **PipelineRoute** - Can be done with ProcessingRoute
2. **HierarchicalRoute** - Not used
3. **All variants** - Move to experimental
4. **RiskServiceRoute** - Replace with ProcessingRoute + RiskProcessor
5. **ExecutionServiceRoute** - Replace with ProcessingRoute + ExecutionProcessor

## Benefits

1. **Clear separation**: Routes handle "where", processors handle "what"
2. **Reusable patterns**: ProcessingRoute works for any validation/transformation
3. **Testable**: Processors can be unit tested without routing infrastructure
4. **Composable**: Can chain processors or swap implementations
5. **Simpler**: Fewer specialized route types to maintain