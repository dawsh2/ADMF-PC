# Adapter Architecture

## Overview

The ADMF-PC adapter architecture provides a pluggable, configuration-driven approach to container communication. Rather than hardcoding event routing logic within containers, adapters externalize and standardize communication patterns, enabling flexible system composition without code changes.

## Core Philosophy

### The Problem with Traditional Routing

Traditional container-based systems often embed routing logic directly in the container:

```python
# Traditional approach - routing logic mixed with business logic
class TradingContainer:
    def _route_events(self, event):
        # This becomes complex as requirements grow
        if self.mode == "backtest":
            if event.type == "SIGNAL":
                self.execution_container.receive(event)
            elif event.type == "RISK_CHECK":
                self.risk_container.receive(event)
        elif self.mode == "optimization":
            if self.phase == "parameter_discovery":
                self.signal_capture.receive(event)
            elif self.phase == "replay":
                self.ensemble_optimizer.receive(event)
        # More conditions accumulate over time...
```

This approach creates several problems:
- **Tight Coupling**: Container logic is mixed with routing logic
- **Poor Testability**: Can't test business logic without routing
- **Limited Flexibility**: Adding new communication patterns requires code changes
- **Maintenance Burden**: Routing logic grows complex with each new requirement

### The Adapter Solution

Adapters separate communication patterns from container implementation:

```python
# Adapter approach - clean separation of concerns
class Container:
    """Container focuses only on business logic"""
    def process(self, event):
        # Pure business logic
        result = self.strategy.calculate(event)
        self.publish(result)  # Adapter handles routing

# Configuration determines routing
adapters:
  - type: "pipeline"
    source: "strategy_container"
    target: "risk_container"
```

## Architecture Components

### 1. Container Protocol

All containers implement a minimal protocol:

```python
from typing import Protocol, Optional, Dict, Any
from abc import abstractmethod

class Container(Protocol):
    """Minimal container interface for adapter compatibility"""
    
    @property
    def name(self) -> str:
        """Unique container identifier"""
        ...
    
    @property
    def event_bus(self) -> EventBus:
        """Container's isolated event bus"""
        ...
    
    def receive_event(self, event: Event) -> None:
        """Receive events from adapters"""
        ...
    
    def publish_event(self, event: Event) -> None:
        """Publish events to adapters"""
        ...
    
    @abstractmethod
    def process(self, event: Event) -> Optional[Event]:
        """Container's business logic"""
        ...
```

### 2. Adapter Protocol

Adapters implement a protocol, not inherit from a base class:

```python
from typing import Protocol, runtime_checkable, Dict, Any
import logging

@runtime_checkable
class CommunicationAdapter(Protocol):
    """Protocol for all communication adapters"""
    
    name: str
    config: Dict[str, Any]
    
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure adapter with container references"""
        ...
    
    def start(self) -> None:
        """Start adapter operation"""
        ...
    
    def stop(self) -> None:
        """Stop adapter operation"""
        ...
    
    def handle_event(self, event: Event, source: Container) -> None:
        """Process event with error handling and metrics"""
        ...

# Helper functions instead of base class methods
def create_adapter_with_logging(adapter_class, name: str, config: Dict[str, Any]):
    """Create adapter with standard infrastructure"""
    adapter = adapter_class(name, config)
    adapter.metrics = AdapterMetrics(name)
    adapter.logger = logging.getLogger(f"adapter.{name}")
    adapter.error_handler = AdapterErrorHandler()
    return adapter

def handle_event_with_metrics(adapter, event: Event, source: Container) -> None:
    """Standard event handling with metrics"""
    with adapter.metrics.measure_latency():
        try:
            adapter.route_event(event, source)
            adapter.metrics.increment_success()
        except Exception as e:
            adapter.metrics.increment_error()
            adapter.error_handler.handle(event, e)
```

### Example Adapter Implementation

```python
class PipelineAdapter:
    """Pipeline adapter - no inheritance needed!"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.containers = config['containers']
        self.connections = []
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure pipeline connections"""
        for i in range(len(self.containers) - 1):
            source = containers[self.containers[i]]
            target = containers[self.containers[i + 1]]
            self.connections.append((source, target))
            
    def start(self) -> None:
        """Start pipeline operation"""
        for source, target in self.connections:
            source.event_bus.subscribe(
                EventType.OUTPUT,
                lambda e, t=target: self.forward_event(e, t)
            )
    
    def stop(self) -> None:
        """Stop pipeline operation"""
        # Unsubscribe from all connections
        pass
    
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event with standard metrics"""
        handle_event_with_metrics(self, event, source)
    
    def route_event(self, event: Event, source: Container) -> None:
        """Route to next container in pipeline"""
        # Find next container and forward
        pass
```

### 3. Adapter Factory

The factory creates adapters from configuration:

```python
class AdapterFactory:
    """Creates adapters from configuration"""
    
    def __init__(self):
        self._registry = {
            'pipeline': PipelineAdapter,
            'broadcast': BroadcastAdapter,
            'hierarchical': HierarchicalAdapter,
            'selective': SelectiveAdapter,
            'composite': CompositeAdapter,
        }
        
    def create(self, config: Dict[str, Any]) -> CommunicationAdapter:
        """Create adapter from configuration"""
        adapter_type = config['type']
        adapter_class = self._registry.get(adapter_type)
        
        if not adapter_class:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
            
        return adapter_class(
            name=config.get('name', f"{adapter_type}_{uuid.uuid4().hex[:8]}"),
            config=config
        )
    
    def register_custom(self, type_name: str, adapter_class: type):
        """Register custom adapter types"""
        self._registry[type_name] = adapter_class
```

### 4. Communication Layer

The communication layer manages all adapters:

```python
class CommunicationLayer:
    """Manages all adapters and their lifecycle"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters: Dict[str, CommunicationAdapter] = {}
        self.containers: Dict[str, Container] = {}
        self._factory = AdapterFactory()
        self._running = False
        
    def add_container(self, container: Container) -> None:
        """Register a container"""
        self.containers[container.name] = container
        container.event_bus.subscribe(
            EventType.ALL, 
            lambda event: self._on_container_event(container, event)
        )
        
    def _on_container_event(self, source: Container, event: Event) -> None:
        """Route events from containers through adapters"""
        for adapter in self.adapters.values():
            if self._should_handle(adapter, source, event):
                adapter.handle_event(event, source)
    
    def _should_handle(self, adapter: CommunicationAdapter, 
                      source: Container, event: Event) -> bool:
        """Determine if adapter should handle this event"""
        # Check if adapter is configured for this source
        if hasattr(adapter, 'source_container'):
            return adapter.source_container == source.name
        elif hasattr(adapter, 'containers'):
            return source.name in adapter.containers
        return False
        
    def setup_adapters(self) -> None:
        """Create and configure all adapters"""
        for adapter_config in self.config.get('adapters', []):
            adapter = self._factory.create(adapter_config)
            adapter.setup(self.containers)
            self.adapters[adapter.name] = adapter
            
    async def start(self) -> None:
        """Start all adapters"""
        self._running = True
        for adapter in self.adapters.values():
            adapter.start()
            self.logger.info(f"Started adapter: {adapter.name}")
            
    async def stop(self) -> None:
        """Stop all adapters"""
        self._running = False
        for adapter in self.adapters.values():
            adapter.stop()
            self.logger.info(f"Stopped adapter: {adapter.name}")
```

## Type Flow Analysis and Validation

The adapter architecture includes a sophisticated type flow analysis system that ensures events flow correctly through the system. This goes beyond simple graph validation to prove that the right event types reach the right containers.

### 1. Type Flow Analyzer

The type flow analyzer tracks which event types can reach which containers:

```python
from typing import Dict, Set, List, Tuple
from enum import Enum

class EventType(Enum):
    """Standard event types in the trading system"""
    BAR = "BAR"
    INDICATOR = "INDICATOR"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    PORTFOLIO_UPDATE = "PORTFOLIO_UPDATE"
    REJECTION = "REJECTION"
    ADJUSTMENT = "ADJUSTMENT"

class FlowNode:
    """Represents a container's type flow information"""
    def __init__(self, name: str):
        self.name = name
        self.can_receive: Set[EventType] = set()  # Types container can handle
        self.can_produce: Set[EventType] = set()  # Types container can emit
        self.will_receive: Set[EventType] = set() # Types that will reach it
        self.will_produce: Set[EventType] = set() # Types it will emit given inputs

class TypeFlowAnalyzer:
    """Analyzes and validates event type flow through adapters"""
    
    def __init__(self):
        # Define canonical event transformations
        self.transformations = {
            'data_source': {
                'produces': {EventType.BAR},
                'consumes': set()
            },
            'indicator_engine': {
                'produces': {EventType.INDICATOR, EventType.BAR},
                'consumes': {EventType.BAR},
                'transforms': {EventType.BAR: {EventType.INDICATOR, EventType.BAR}}
            },
            'strategy': {
                'produces': {EventType.SIGNAL},
                'consumes': {EventType.BAR, EventType.INDICATOR},
                'transforms': {
                    EventType.BAR: {EventType.SIGNAL},
                    EventType.INDICATOR: {EventType.SIGNAL}
                }
            },
            'risk_manager': {
                'produces': {EventType.ORDER, EventType.REJECTION},
                'consumes': {EventType.SIGNAL},
                'transforms': {EventType.SIGNAL: {EventType.ORDER, EventType.REJECTION}}
            },
            'execution_engine': {
                'produces': {EventType.FILL},
                'consumes': {EventType.ORDER},
                'transforms': {EventType.ORDER: {EventType.FILL}}
            },
            'portfolio_manager': {
                'produces': {EventType.PORTFOLIO_UPDATE},
                'consumes': {EventType.FILL},
                'transforms': {EventType.FILL: {EventType.PORTFOLIO_UPDATE}}
            }
        }
        
        # Define valid execution modes and their requirements
        self.execution_modes = {
            'full_backtest': {
                'required_flows': [
                    (EventType.BAR, 'data_source', 'strategy'),
                    (EventType.SIGNAL, 'strategy', 'risk_manager'),
                    (EventType.ORDER, 'risk_manager', 'execution_engine'),
                    (EventType.FILL, 'execution_engine', 'portfolio_manager')
                ],
                'must_produce': {EventType.PORTFOLIO_UPDATE}
            },
            'signal_generation': {
                'required_flows': [
                    (EventType.BAR, 'data_source', 'strategy')
                ],
                'must_produce': {EventType.SIGNAL}
            },
            'signal_replay': {
                'required_flows': [
                    (EventType.SIGNAL, 'signal_source', 'risk_manager'),
                    (EventType.ORDER, 'risk_manager', 'execution_engine')
                ],
                'must_produce': {EventType.PORTFOLIO_UPDATE}
            }
        }
    
    def analyze_flow(self, containers: Dict[str, Container], 
                    adapters: List[CommunicationAdapter]) -> Dict[str, FlowNode]:
        """Build complete type flow map from adapters and containers"""
        flow_map = {}
        
        # Initialize flow nodes from containers
        for name, container in containers.items():
            node = FlowNode(name)
            container_type = self._get_container_type(container)
            
            if container_type in self.transformations:
                node.can_receive = self.transformations[container_type]['consumes']
                node.can_produce = self.transformations[container_type]['produces']
            
            flow_map[name] = node
        
        # Propagate types through adapters
        self._propagate_types(flow_map, adapters)
        
        return flow_map
    
    def _propagate_types(self, flow_map: Dict[str, FlowNode], 
                        adapters: List[CommunicationAdapter]) -> None:
        """Propagate event types through adapter connections"""
        # Build adjacency list from adapters
        connections = self._build_connections(adapters)
        
        # Iteratively propagate types until no changes
        changed = True
        iterations = 0
        max_iterations = 100  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for source_name, targets in connections.items():
                source_node = flow_map.get(source_name)
                if not source_node:
                    continue
                    
                # Determine what types this source will produce
                produced_types = self._compute_produced_types(
                    source_node, source_name
                )
                
                # Propagate to all targets
                for target_name in targets:
                    target_node = flow_map.get(target_name)
                    if not target_node:
                        continue
                        
                    # Add types that target will receive
                    before_size = len(target_node.will_receive)
                    target_node.will_receive.update(
                        produced_types & target_node.can_receive
                    )
                    
                    if len(target_node.will_receive) > before_size:
                        changed = True
    
    def _compute_produced_types(self, node: FlowNode, 
                               container_name: str) -> Set[EventType]:
        """Compute what types a container will produce given its inputs"""
        container_type = self._infer_container_type(container_name)
        
        if container_type not in self.transformations:
            return node.can_produce
            
        transforms = self.transformations[container_type].get('transforms', {})
        produced = set()
        
        # Apply transformations based on what the container will receive
        for input_type in node.will_receive:
            if input_type in transforms:
                produced.update(transforms[input_type])
            else:
                # Pass through if no transformation defined
                if input_type in node.can_produce:
                    produced.add(input_type)
        
        # Always include static productions (e.g., data sources)
        if not node.will_receive and node.can_produce:
            produced.update(node.can_produce)
            
        return produced
    
    def validate(self, flow_map: Dict[str, FlowNode], 
                mode: str) -> ValidationResult:
        """Validate that type flow meets requirements for execution mode"""
        if mode not in self.execution_modes:
            return ValidationResult(
                valid=False,
                error=f"Unknown execution mode: {mode}"
            )
        
        mode_config = self.execution_modes[mode]
        errors = []
        warnings = []
        
        # Check required flows
        for event_type, source_role, target_role in mode_config['required_flows']:
            if not self._check_flow_exists(flow_map, event_type, 
                                         source_role, target_role):
                errors.append(
                    f"Missing required flow: {event_type} from "
                    f"{source_role} to {target_role}"
                )
        
        # Check required productions
        for required_type in mode_config['must_produce']:
            if not self._check_type_produced(flow_map, required_type):
                errors.append(
                    f"No container produces required event type: {required_type}"
                )
        
        # Check for type mismatches
        mismatches = self._check_type_mismatches(flow_map)
        errors.extend(mismatches)
        
        # Check for orphaned events
        orphans = self._check_orphaned_events(flow_map)
        warnings.extend(orphans)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            flow_map=flow_map
        )
    
    def _check_flow_exists(self, flow_map: Dict[str, FlowNode],
                          event_type: EventType,
                          source_role: str, 
                          target_role: str) -> bool:
        """Check if a specific event type flows from source to target role"""
        # Find containers matching roles
        sources = [n for n in flow_map.values() 
                  if self._matches_role(n.name, source_role)]
        targets = [n for n in flow_map.values() 
                  if self._matches_role(n.name, target_role)]
        
        if not sources or not targets:
            return False
            
        # Check if any source produces the type and any target receives it
        for source in sources:
            produced = self._compute_produced_types(source, source.name)
            if event_type in produced:
                for target in targets:
                    if event_type in target.will_receive:
                        return True
        
        return False
    
    def _build_connections(self, 
                          adapters: List[CommunicationAdapter]) -> Dict[str, List[str]]:
        """Build adjacency list from adapter configurations"""
        connections = {}
        
        for adapter in adapters:
            if hasattr(adapter, 'source') and hasattr(adapter, 'targets'):
                # Broadcast pattern
                connections.setdefault(adapter.source, []).extend(adapter.targets)
            elif hasattr(adapter, 'containers'):
                # Pipeline pattern
                containers = adapter.containers
                for i in range(len(containers) - 1):
                    connections.setdefault(containers[i], []).append(containers[i + 1])
            elif hasattr(adapter, 'parent') and hasattr(adapter, 'children'):
                # Hierarchical pattern
                for child in adapter.children:
                    child_name = child['name'] if isinstance(child, dict) else child
                    connections.setdefault(adapter.parent, []).append(child_name)
        
        return connections
```

### 2. Visualization Support

The type flow analyzer includes visualization capabilities for debugging:

```python
class TypeFlowVisualizer:
    """Visualize type flow through the system"""
    
    def __init__(self, analyzer: TypeFlowAnalyzer):
        self.analyzer = analyzer
    
    def generate_text_visualization(self, 
                                  flow_map: Dict[str, FlowNode]) -> str:
        """Generate human-readable flow visualization"""
        lines = ["Type Flow Analysis:", "=" * 50, ""]
        
        # Sort containers by typical flow order
        ordered_containers = self._order_by_flow(flow_map)
        
        for container_name in ordered_containers:
            node = flow_map[container_name]
            lines.append(f"{container_name}:")
            
            if node.will_receive:
                lines.append(f"  receives: {self._format_types(node.will_receive)}")
            else:
                lines.append("  receives: [none - source]")
                
            produced = self.analyzer._compute_produced_types(node, container_name)
            if produced:
                lines.append(f"  produces: {self._format_types(produced)}")
            else:
                lines.append("  produces: [none]")
                
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_mermaid_diagram(self, 
                                flow_map: Dict[str, FlowNode],
                                adapters: List[CommunicationAdapter]) -> str:
        """Generate Mermaid diagram for visualization"""
        lines = ["graph TD"]
        connections = self.analyzer._build_connections(adapters)
        
        # Add nodes with their event types
        for name, node in flow_map.items():
            produced = self.analyzer._compute_produced_types(node, name)
            if produced:
                label = f"{name}<br/>[{self._format_types(produced)}]"
            else:
                label = name
            lines.append(f"    {name}[{label}]")
        
        # Add edges with event types
        for source, targets in connections.items():
            source_node = flow_map.get(source)
            if not source_node:
                continue
                
            produced = self.analyzer._compute_produced_types(source_node, source)
            
            for target in targets:
                target_node = flow_map.get(target)
                if not target_node:
                    continue
                    
                # Find what types actually flow
                flowing_types = produced & target_node.can_receive
                if flowing_types:
                    label = self._format_types(flowing_types)
                    lines.append(f"    {source} -->|{label}| {target}")
                else:
                    lines.append(f"    {source} -.->|no match| {target}")
        
        return "\n".join(lines)
    
    def generate_validation_report(self, 
                                 validation_result: ValidationResult) -> str:
        """Generate detailed validation report"""
        lines = ["Type Flow Validation Report", "=" * 50, ""]
        
        if validation_result.valid:
            lines.append("✓ Type flow validation PASSED")
        else:
            lines.append("✗ Type flow validation FAILED")
        
        if validation_result.errors:
            lines.extend(["", "Errors:", "-" * 20])
            for error in validation_result.errors:
                lines.append(f"  • {error}")
        
        if validation_result.warnings:
            lines.extend(["", "Warnings:", "-" * 20])
            for warning in validation_result.warnings:
                lines.append(f"  • {warning}")
        
        if validation_result.flow_map:
            lines.extend(["", "", self.generate_text_visualization(
                validation_result.flow_map
            )])
        
        return "\n".join(lines)
    
    def _format_types(self, types: Set[EventType]) -> str:
        """Format event types for display"""
        return ", ".join(sorted(t.value for t in types))
    
    def _order_by_flow(self, flow_map: Dict[str, FlowNode]) -> List[str]:
        """Order containers by typical flow sequence"""
        # Simple heuristic ordering
        order_priority = {
            'data': 0, 'indicator': 1, 'strategy': 2,
            'risk': 3, 'execution': 4, 'portfolio': 5
        }
        
        def priority(name: str) -> int:
            for key, pri in order_priority.items():
                if key in name.lower():
                    return pri
            return 999
        
        return sorted(flow_map.keys(), key=priority)
```

### 3. Integration with Semantic Events

Type flow analysis integrates with semantic events for even stronger validation:

```python
class SemanticTypeFlowAnalyzer(TypeFlowAnalyzer):
    """Extended analyzer for semantic event validation"""
    
    def validate_semantic_compatibility(self, 
                                      flow_map: Dict[str, FlowNode],
                                      semantic_registry: Dict[str, SemanticEventSchema]) -> List[str]:
        """Validate semantic event schema compatibility"""
        errors = []
        
        # Check each connection for schema compatibility
        for source_name, node in flow_map.items():
            produced_types = self._compute_produced_types(node, source_name)
            
            for event_type in produced_types:
                schema = semantic_registry.get(event_type.value)
                if not schema:
                    continue
                    
                # Find all consumers of this event type
                consumers = self._find_consumers(flow_map, event_type)
                
                for consumer_name in consumers:
                    consumer_schema = self._get_expected_schema(
                        consumer_name, event_type, semantic_registry
                    )
                    
                    if not self._schemas_compatible(schema, consumer_schema):
                        errors.append(
                            f"{source_name} produces {event_type} with schema "
                            f"{schema.version}, but {consumer_name} expects "
                            f"{consumer_schema.version}"
                        )
        
        return errors
```

### 4. Usage Example

```python
# Create analyzer
analyzer = TypeFlowAnalyzer()
visualizer = TypeFlowVisualizer(analyzer)

# Analyze adapter configuration
flow_map = analyzer.analyze_flow(containers, adapters)

# Validate for specific execution mode
validation = analyzer.validate(flow_map, 'full_backtest')

if not validation.valid:
    print(visualizer.generate_validation_report(validation))
    raise ConfigurationError("Invalid adapter configuration")

# Generate visualization for debugging
print(visualizer.generate_text_visualization(flow_map))

# Generate Mermaid diagram
with open('type_flow.mmd', 'w') as f:
    f.write(visualizer.generate_mermaid_diagram(flow_map, adapters))
```

The type flow analysis system ensures that:
- Events flow through valid paths from source to destination
- Required event transformations occur
- No type mismatches between producers and consumers
- All execution modes have their required flows
- Semantic event schemas are compatible across connections

This provides much stronger guarantees than simple graph validation, catching configuration errors before runtime.

## Performance Considerations

### 1. Event Batching

Adapters can batch events for efficiency:

```python
class BatchingMixin:
    """Mixin for adapters that support batching"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = self.config.get('batch_size', 100)
        self.batch_timeout = self.config.get('batch_timeout_ms', 10)
        self._batch: List[Event] = []
        self._batch_timer = None
        
    def add_to_batch(self, event: Event) -> None:
        """Add event to batch"""
        self._batch.append(event)
        
        if len(self._batch) >= self.batch_size:
            self._flush_batch()
        elif not self._batch_timer:
            self._start_batch_timer()
            
    def _flush_batch(self) -> None:
        """Process accumulated batch"""
        if self._batch:
            self._process_batch(self._batch)
            self._batch = []
            self._cancel_batch_timer()
            
    @abstractmethod
    def _process_batch(self, events: List[Event]) -> None:
        """Process a batch of events"""
        pass
```

### 2. Performance Tiers

Different adapters operate at different performance tiers:

```python
class PerformanceTier(Enum):
    """Performance characteristics for adapters"""
    
    FAST = {
        'max_latency_ms': 1,
        'batch_size': 1000,
        'delivery_guarantee': 'at_most_once',
        'serialization': 'zero_copy'
    }
    
    STANDARD = {
        'max_latency_ms': 10,
        'batch_size': 100,
        'delivery_guarantee': 'at_least_once',
        'serialization': 'efficient'
    }
    
    RELIABLE = {
        'max_latency_ms': 100,
        'batch_size': 1,
        'delivery_guarantee': 'exactly_once',
        'serialization': 'safe',
        'persistence': True
    }
```

### 3. Zero-Copy Optimization

For high-frequency events, adapters can use zero-copy techniques:

```python
class ZeroCopyAdapter(CommunicationAdapter):
    """Adapter optimized for zero-copy event passing"""
    
    def _route_event(self, event: Event, source: Container) -> None:
        """Route event without copying"""
        if isinstance(event, ZeroCopyEvent):
            # Pass reference directly
            self.target.receive_event_ref(event)
        else:
            # Fall back to normal routing
            super()._route_event(event, source)
```

## Error Handling

### 1. Error Handler

Centralized error handling for all adapters:

```python
class AdapterErrorHandler:
    """Handles errors in adapter operations"""
    
    def __init__(self):
        self.retry_policy = ExponentialBackoff()
        self.circuit_breaker = CircuitBreaker()
        self.dead_letter_queue = DeadLetterQueue()
        
    def handle(self, event: Event, error: Exception) -> None:
        """Handle adapter errors"""
        error_context = ErrorContext(event, error)
        
        # Check circuit breaker
        if self.circuit_breaker.is_open(error_context.target):
            self.dead_letter_queue.add(error_context)
            return
            
        # Attempt retry
        if self.retry_policy.should_retry(error_context):
            self.retry_policy.schedule_retry(error_context)
        else:
            self.dead_letter_queue.add(error_context)
            self.circuit_breaker.record_failure(error_context.target)
```

### 2. Graceful Degradation

Adapters can degrade gracefully under load:

```python
class AdaptiveThroughputControl:
    """Dynamically adjusts throughput based on system load"""
    
    def __init__(self, adapter: CommunicationAdapter):
        self.adapter = adapter
        self.current_rate = 1.0
        self.target_latency_ms = 10
        
    def adjust_rate(self, current_latency_ms: float) -> None:
        """Adjust processing rate based on latency"""
        if current_latency_ms > self.target_latency_ms * 1.5:
            # Reduce rate
            self.current_rate *= 0.9
        elif current_latency_ms < self.target_latency_ms * 0.5:
            # Increase rate
            self.current_rate = min(1.0, self.current_rate * 1.1)
            
        self.adapter.set_throughput_limit(self.current_rate)
```

## Monitoring and Observability

### 1. Adapter Metrics

Each adapter tracks comprehensive metrics:

```python
class AdapterMetrics:
    """Metrics collection for adapters"""
    
    def __init__(self, adapter_name: str):
        self.adapter_name = adapter_name
        self.events_processed = Counter(
            'adapter_events_processed',
            'Total events processed',
            ['adapter', 'status']
        )
        self.processing_latency = Histogram(
            'adapter_processing_latency_seconds',
            'Event processing latency',
            ['adapter']
        )
        self.active_connections = Gauge(
            'adapter_active_connections',
            'Number of active connections',
            ['adapter']
        )
        
    @contextmanager
    def measure_latency(self):
        """Context manager for measuring latency"""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.processing_latency.labels(
                adapter=self.adapter_name
            ).observe(duration)
```

### 2. Health Checks

Adapters provide health check endpoints:

```python
class AdapterHealthCheck:
    """Health check for adapters"""
    
    def __init__(self, adapter: CommunicationAdapter):
        self.adapter = adapter
        
    def check_health(self) -> HealthStatus:
        """Perform health check"""
        checks = [
            self._check_connections(),
            self._check_throughput(),
            self._check_error_rate(),
            self._check_latency()
        ]
        
        if all(check.is_healthy for check in checks):
            return HealthStatus.HEALTHY
        elif any(check.is_critical for check in checks):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED
```

## Configuration Examples

### Basic Pipeline Configuration

```yaml
adapters:
  - type: "pipeline"
    name: "main_flow"
    containers: ["data_reader", "strategy", "risk_manager", "executor"]
    tier: "standard"
    error_handling:
      retry_attempts: 3
      retry_delay_ms: 100
```

### Complex Multi-Pattern Configuration

```yaml
adapters:
  # Data distribution
  - type: "broadcast"
    name: "market_data_distribution"
    source: "data_reader"
    targets: ["strategy_a", "strategy_b", "strategy_c"]
    tier: "fast"
    batch_size: 1000
    
  # Hierarchical risk management
  - type: "hierarchical"
    name: "risk_hierarchy"
    parent: "portfolio_risk_manager"
    children:
      - container: "strategy_a_risk"
        weight: 0.4
      - container: "strategy_b_risk"
        weight: 0.6
    context_propagation: true
    
  # Conditional routing
  - type: "selective"
    name: "signal_router"
    source: "signal_aggregator"
    rules:
      - condition: "signal.confidence > 0.8"
        target: "aggressive_executor"
      - condition: "signal.confidence > 0.5"
        target: "normal_executor"
      - condition: "default"
        target: "conservative_executor"
```

## Testing Adapters

### 1. Unit Testing

Test adapters in isolation:

```python
class TestPipelineAdapter:
    """Unit tests for pipeline adapter"""
    
    def test_event_routing(self):
        """Test basic event routing"""
        # Create mock containers
        source = MockContainer("source")
        target = MockContainer("target")
        
        # Create adapter
        adapter = PipelineAdapter(
            name="test_pipeline",
            config={"containers": ["source", "target"]}
        )
        adapter.setup({"source": source, "target": target})
        
        # Send event
        event = TestEvent(data="test")
        adapter.handle_event(event, source)
        
        # Verify routing
        assert target.received_events[-1] == event
```

### 2. Integration Testing

Test adapter interactions:

```python
class TestAdapterIntegration:
    """Integration tests for adapter system"""
    
    async def test_multi_adapter_flow(self):
        """Test multiple adapters working together"""
        # Create communication layer
        comm_layer = CommunicationLayer({
            'adapters': [
                {'type': 'broadcast', 'source': 'data', 'targets': ['s1', 's2']},
                {'type': 'pipeline', 'containers': ['s1', 'risk', 'exec']},
                {'type': 'pipeline', 'containers': ['s2', 'risk', 'exec']}
            ]
        })
        
        # Add containers
        containers = create_test_containers()
        for container in containers:
            comm_layer.add_container(container)
            
        # Setup and start
        comm_layer.setup_adapters()
        await comm_layer.start()
        
        # Send test event
        data_container = containers['data']
        data_container.publish_event(MarketDataEvent(symbol="AAPL", price=150.0))
        
        # Verify event flow
        await assert_event_flow_completed(containers)
```

## Best Practices

### 1. Adapter Design Guidelines

- **Single Responsibility**: Each adapter should handle one communication pattern
- **Configuration Over Code**: Behavior should be configurable, not hardcoded
- **Fail Fast**: Detect configuration errors during setup, not runtime
- **Observable**: Comprehensive metrics and logging

### 2. Performance Guidelines

- **Choose Appropriate Tiers**: Match adapter tier to event characteristics
- **Enable Batching**: For high-frequency events, always enable batching
- **Monitor Latency**: Set up alerts for latency violations
- **Plan for Failure**: Implement circuit breakers and fallbacks

### 3. Testing Guidelines

- **Test in Isolation**: Unit test each adapter independently
- **Test Error Paths**: Verify error handling and recovery
- **Load Test**: Ensure adapters perform under expected load
- **Test Configuration**: Validate all configuration options

## Conclusion

The adapter architecture provides a flexible, maintainable approach to container communication. By externalizing routing logic into pluggable adapters, the system gains:

1. **Flexibility**: Change communication patterns without code changes
2. **Maintainability**: Clear separation between business logic and communication
3. **Testability**: Test containers and adapters independently
4. **Performance**: Optimized adapters for different event characteristics
5. **Reliability**: Centralized error handling and monitoring

The key insight is that communication patterns are a separate concern from business logic and should be treated as such in the architecture.
