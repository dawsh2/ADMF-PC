# Flexible Event Communication Architecture

## The Core Problem

We need to maintain the benefits of isolated event buses (resource isolation, parallelization, reproducibility) while supporting different container organization patterns (Strategy-First, Classifier-First, Risk-First, Portfolio-First) without forcing tight coupling between organizational structure and event communication.

## Solution: Pluggable Event Communication Adapters

Instead of hardcoding event flows based on organizational patterns, use **Event Communication Adapters** that can be configured independently of container organization.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTAINER ORGANIZATION                        │
│  (Strategy-First, Classifier-First, Risk-First, Portfolio-First) │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            EVENT COMMUNICATION LAYER                        │ │
│  │                                                             │ │
│  │  ┌─────────────────┐    ┌─────────────────────────────────┐ │ │
│  │  │ Communication   │    │ Event Flow Patterns             │ │ │
│  │  │ Adapter Factory │────│ • Pipeline                      │ │ │
│  │  │                 │    │ • Hierarchical                  │ │ │
│  │  └─────────────────┘    │ • Broadcast                     │ │ │
│  │                         │ • Selective                     │ │ │
│  │                         └─────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              ISOLATED EVENT BUSES                           │ │
│  │  (Per container, maintaining all isolation benefits)        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Event Communication Patterns

### 1. Pipeline Pattern (Default for Most Cases)
**Best for**: Linear data transformation workflows

```python
class PipelineCommunicationAdapter:
    """Simple linear event transformation between containers"""
    
    def setup_flow(self, containers: List[Container]) -> None:
        """Set up linear pipeline flow"""
        for i, container in enumerate(containers[:-1]):
            next_container = containers[i + 1]
            
            # Set up transformation mapping
            container.on_output_event(
                lambda event: self.transform_and_forward(event, next_container)
            )
    
    def transform_and_forward(self, event: Event, target: Container) -> None:
        """Transform event and forward to next stage"""
        # Transform event based on type mapping
        transformed_event = self.event_transformer.transform(event)
        target.receive_event(transformed_event)

# Usage with any organizational pattern
pipeline_adapter = PipelineCommunicationAdapter()
pipeline_adapter.setup_flow([data_container, strategy_container, risk_container, execution_container])
```

### 2. Hierarchical Pattern (For Classifier-First)
**Best for**: Parent-child relationships with regime context

```python
class HierarchicalCommunicationAdapter:
    """Parent-child event forwarding with context preservation"""
    
    def setup_hierarchy(self, parent: Container, children: List[Container]) -> None:
        """Set up parent-to-children event forwarding"""
        
        # Parent events flow down to all children
        parent.on_context_event(
            lambda event: self.broadcast_to_children(event, children)
        )
        
        # Children events aggregate up to parent
        for child in children:
            child.on_result_event(
                lambda event: self.aggregate_to_parent(event, parent)
            )
    
    def broadcast_to_children(self, event: Event, children: List[Container]) -> None:
        """Send context events to all children"""
        for child in children:
            child.receive_context_event(event)
    
    def aggregate_to_parent(self, event: Event, parent: Container) -> None:
        """Aggregate child results to parent"""
        parent.receive_child_result(event)

# Usage
hierarchical_adapter = HierarchicalCommunicationAdapter()
hierarchical_adapter.setup_hierarchy(classifier_container, [risk_container_a, risk_container_b])
```

### 3. Broadcast Pattern (For Multi-Strategy)
**Best for**: One-to-many communication

```python
class BroadcastCommunicationAdapter:
    """Broadcast events to multiple consumers"""
    
    def setup_broadcast(self, source: Container, targets: List[Container]) -> None:
        """Set up one-to-many broadcast"""
        
        source.on_broadcast_event(
            lambda event: self.broadcast_to_targets(event, targets)
        )
    
    def broadcast_to_targets(self, event: Event, targets: List[Container]) -> None:
        """Send event to all targets"""
        for target in targets:
            # Each target gets its own copy to maintain isolation
            cloned_event = event.clone()
            target.receive_event(cloned_event)

# Usage
broadcast_adapter = BroadcastCommunicationAdapter()
broadcast_adapter.setup_broadcast(data_container, [strategy_a, strategy_b, strategy_c])
```

### 4. Selective Pattern (For Complex Routing)
**Best for**: Conditional event routing based on content

```python
class SelectiveCommunicationAdapter:
    """Route events based on content and rules"""
    
    def __init__(self):
        self.routing_rules = {}
    
    def add_routing_rule(self, condition: Callable, target: Container) -> None:
        """Add conditional routing rule"""
        self.routing_rules[condition] = target
    
    def setup_selective_routing(self, source: Container) -> None:
        """Set up content-based routing"""
        
        source.on_event(
            lambda event: self.route_event(event)
        )
    
    def route_event(self, event: Event) -> None:
        """Route event based on rules"""
        for condition, target in self.routing_rules.items():
            if condition(event):
                target.receive_event(event)
                break

# Usage
selective_adapter = SelectiveCommunicationAdapter()
selective_adapter.add_routing_rule(
    lambda event: event.regime == "BULL", 
    aggressive_risk_container
)
selective_adapter.add_routing_rule(
    lambda event: event.regime == "BEAR", 
    conservative_risk_container
)
selective_adapter.setup_selective_routing(classifier_container)
```

## Configuration-Driven Communication

### YAML Configuration Example

```yaml
# Event communication is configured separately from container organization
event_communication:
  pattern: "flexible"  # Use multiple adapters
  
  adapters:
    - type: "pipeline"
      name: "main_data_flow"
      containers: ["data", "indicators", "strategies", "risk", "execution"]
      
    - type: "hierarchical"
      name: "classifier_hierarchy"
      parent: "hmm_classifier"
      children: ["conservative_risk", "balanced_risk", "aggressive_risk"]
      
    - type: "broadcast"
      name: "indicator_distribution"
      source: "indicator_hub"
      targets: ["strategy_a", "strategy_b", "strategy_c"]
      
    - type: "selective"
      name: "regime_routing"
      source: "classifier"
      rules:
        - condition: "event.regime == 'BULL'"
          target: "aggressive_portfolio"
        - condition: "event.regime == 'BEAR'"
          target: "defensive_portfolio"
        - condition: "event.regime == 'NEUTRAL'"
          target: "balanced_portfolio"

# Container organization configured separately
organization: "classifier_first"
classifiers:
  - name: "hmm_classifier"
    # ... classifier config
```

## Adapter Factory Pattern

```python
class EventCommunicationFactory:
    """Factory to create appropriate communication adapters"""
    
    def __init__(self):
        self.adapters = {
            'pipeline': PipelineCommunicationAdapter,
            'hierarchical': HierarchicalCommunicationAdapter,
            'broadcast': BroadcastCommunicationAdapter,
            'selective': SelectiveCommunicationAdapter
        }
    
    def create_communication_layer(self, config: Dict, containers: Dict[str, Container]) -> CommunicationLayer:
        """Create communication layer from config"""
        
        communication_layer = CommunicationLayer()
        
        for adapter_config in config.get('adapters', []):
            adapter_type = adapter_config['type']
            adapter_class = self.adapters[adapter_type]
            adapter = adapter_class()
            
            # Configure adapter based on type
            self.configure_adapter(adapter, adapter_config, containers)
            communication_layer.add_adapter(adapter)
        
        return communication_layer
    
    def configure_adapter(self, adapter, config: Dict, containers: Dict[str, Container]) -> None:
        """Configure specific adapter based on its type"""
        
        if isinstance(adapter, PipelineCommunicationAdapter):
            container_names = config['containers']
            container_list = [containers[name] for name in container_names]
            adapter.setup_flow(container_list)
            
        elif isinstance(adapter, HierarchicalCommunicationAdapter):
            parent = containers[config['parent']]
            children = [containers[name] for name in config['children']]
            adapter.setup_hierarchy(parent, children)
            
        elif isinstance(adapter, BroadcastCommunicationAdapter):
            source = containers[config['source']]
            targets = [containers[name] for name in config['targets']]
            adapter.setup_broadcast(source, targets)
            
        elif isinstance(adapter, SelectiveCommunicationAdapter):
            source = containers[config['source']]
            for rule in config['rules']:
                condition = self.parse_condition(rule['condition'])
                target = containers[rule['target']]
                adapter.add_routing_rule(condition, target)
            adapter.setup_selective_routing(source)
```

## Benefits of This Approach

### 1. Organizational Pattern Independence
```yaml
# Same communication setup works with any organizational pattern
event_communication:
  adapters:
    - type: "pipeline"
      containers: ["momentum_strategy", "risk_manager", "execution"]

# Works with strategy-first
organization: "strategy_first"
strategies:
  - name: "momentum_strategy"
    # ...

# Also works with classifier-first
organization: "classifier_first"
classifiers:
  - name: "regime_detector"
    risk_profiles:
      - portfolios:
          - strategies:
              - name: "momentum_strategy"
                # ...
```

### 2. Maintains Isolation Benefits
- **Each container still has its own isolated event bus**
- **Inter-container communication is explicit and configurable**
- **Resource isolation preserved for parallelization**
- **No shared state or event bus contamination**

### 3. Flexible Communication Patterns
- **Mix and match adapters as needed**
- **Add new communication patterns without changing containers**
- **Configure different patterns for different phases (backtest vs optimization)**

### 4. Easy Testing and Debugging
```python
# Test with mock communication adapter
mock_adapter = MockCommunicationAdapter()
mock_adapter.record_all_events()

# Run test
communication_layer = CommunicationLayer()
communication_layer.add_adapter(mock_adapter)
container.process_data()

# Verify communication
assert len(mock_adapter.recorded_events) == expected_count
assert mock_adapter.recorded_events[0].type == "SIGNAL"
```

## Default Communication Patterns by Organization

### Strategy-First Default
```yaml
event_communication:
  adapters:
    - type: "pipeline"
      containers: ["data", "strategy_a", "execution"]
    - type: "pipeline"  
      containers: ["data", "strategy_b", "execution"]
    - type: "broadcast"
      source: "execution"
      targets: ["performance_tracker"]
```

### Classifier-First Default
```yaml
event_communication:
  adapters:
    - type: "pipeline"
      containers: ["data", "indicators", "classifier"]
    - type: "hierarchical"
      parent: "classifier"
      children: ["risk_conservative", "risk_aggressive"]
    - type: "pipeline"
      containers: ["risk_conservative", "execution"]
    - type: "pipeline"
      containers: ["risk_aggressive", "execution"]
```

### Risk-First Default
```yaml
event_communication:
  adapters:
    - type: "broadcast"
      source: "data"
      targets: ["strategy_a", "strategy_b"]
    - type: "pipeline"
      containers: ["strategy_a", "risk_manager", "execution"]
    - type: "pipeline"
      containers: ["strategy_b", "risk_manager", "execution"]
```

## Implementation Strategy

### Phase 1: Basic Adapters
- Implement Pipeline and Broadcast adapters
- Create configuration system
- Migrate existing isolated buses to use adapters

### Phase 2: Advanced Patterns
- Add Hierarchical and Selective adapters
- Implement adapter factory and auto-configuration
- Add monitoring and debugging tools

### Phase 3: Optimization
- Add performance monitoring for communication patterns
- Implement adapter performance optimizations
- Add adaptive communication pattern selection

## Conclusion

This approach provides:

1. **Flexibility**: Communication patterns independent of organizational patterns
2. **Isolation**: Maintains all benefits of isolated event buses
3. **Configurability**: Easy to change communication without code changes
4. **Testability**: Mock adapters for testing
5. **Performance**: Optimized patterns for different use cases
6. **Scalability**: Works with massive parallelization

The key insight is that **event communication is orthogonal to container organization** - they solve different problems and should be configurable independently.