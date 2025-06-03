# Adapter Patterns

## Overview

This document describes common adapter patterns in ADMF-PC and their use cases. Each pattern addresses specific communication needs while maintaining the benefits of isolated containers and event-driven architecture.

## Core Adapter Patterns

### 1. Pipeline Pattern

The pipeline pattern creates a linear flow of events through a sequence of containers.

#### Use Cases
- **Data Processing Pipelines**: Data → Indicators → Strategy → Risk → Execution
- **Sequential Workflows**: Each stage depends on the previous one
- **ETL Operations**: Extract → Transform → Load sequences

#### Implementation

```python
class PipelineAdapter:
    """Sequential event flow through containers"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.containers = config['containers']
        self.allow_skip = config.get('allow_skip', False)
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure pipeline connections"""
        for i in range(len(self.containers) - 1):
            source_name = self.containers[i]
            target_name = self.containers[i + 1]
            
            source = containers[source_name]
            target = containers[target_name]
            
            # Subscribe to source events
            source.event_bus.subscribe(
                EventType.OUTPUT,
                lambda event, t=target: self.forward(event, t)
            )
    
    def forward(self, event: Event, target: Container) -> None:
        """Forward event to next container in pipeline"""
        if self.allow_skip and event.metadata.get('skip_stage'):
            # Skip this stage if requested
            self.forward_to_next(event, target)
        else:
            target.receive_event(event)
            
    def start(self) -> None:
        """Start adapter operation"""
        pass
        
    def stop(self) -> None:
        """Stop adapter operation"""
        pass
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event - for protocol compatibility"""
        # Pipeline handles events through subscriptions
        pass
```

#### Configuration Example

```yaml
adapters:
  - type: "pipeline"
    name: "trading_pipeline"
    containers: 
      - "market_data"
      - "indicators"
      - "strategy"
      - "risk_manager"
      - "executor"
    tier: "standard"
    allow_skip: false
```

#### Variations

**Conditional Pipeline**: Skip stages based on event content
```yaml
adapters:
  - type: "pipeline"
    name: "conditional_pipeline"
    containers: ["data", "strategy", "risk", "executor"]
    conditions:
      - stage: "risk"
        skip_if: "event.risk_override == True"
```

**Parallel Pipeline**: Process through multiple pipelines simultaneously
```yaml
adapters:
  - type: "parallel_pipeline"
    name: "multi_strategy_pipeline"
    pipelines:
      - ["data", "momentum_strategy", "executor"]
      - ["data", "mean_reversion_strategy", "executor"]
    merge_at: "portfolio_manager"
```

### 2. Broadcast Pattern

The broadcast pattern distributes events from one source to multiple targets.

#### Use Cases
- **Market Data Distribution**: Send price updates to all strategies
- **Alert Broadcasting**: Notify multiple risk managers
- **Event Fanout**: Distribute signals to multiple consumers

#### Implementation

```python
class BroadcastAdapter:
    """One-to-many event distribution"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.source = config['source']
        self.targets = config['targets']
        self.filter_empty = config.get('filter_empty', True)
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure broadcast connections"""
        source = containers[self.source]
        self.target_containers = [
            containers[name] for name in self.targets
        ]
        
        source.event_bus.subscribe(
            EventType.OUTPUT,
            self.broadcast
        )
    
    def broadcast(self, event: Event) -> None:
        """Broadcast event to all targets"""
        if self.filter_empty and not event.data:
            return
            
        for target in self.target_containers:
            # Clone event for each target
            cloned_event = self.clone_event(event)
            target.receive_event(cloned_event)
            
    def clone_event(self, event: Event) -> Event:
        """Create a copy of the event"""
        return copy.deepcopy(event)
        
    def start(self) -> None:
        """Start adapter operation"""
        pass
        
    def stop(self) -> None:
        """Stop adapter operation"""
        pass
        
    def handle_event(self, event: Event, source: Container) -> None:
        """Handle event - for protocol compatibility"""
        # Broadcast handles events through subscriptions
        pass
```

#### Configuration Example

```yaml
adapters:
  - type: "broadcast"
    name: "market_data_fanout"
    source: "market_data_reader"
    targets:
      - "momentum_strategy"
      - "mean_reversion_strategy"
      - "arbitrage_strategy"
      - "market_maker"
    tier: "fast"
    filter_empty: true
```

#### Variations

**Filtered Broadcast**: Only send to targets that match criteria
```yaml
adapters:
  - type: "filtered_broadcast"
    name: "selective_distribution"
    source: "signal_generator"
    targets:
      - name: "high_freq_strategy"
        filter: "event.frequency == 'high'"
      - name: "low_freq_strategy"
        filter: "event.frequency == 'low'"
```

**Load-Balanced Broadcast**: Distribute events across targets
```yaml
adapters:
  - type: "load_balanced_broadcast"
    name: "work_distribution"
    source: "task_generator"
    targets: ["worker_1", "worker_2", "worker_3"]
    strategy: "round_robin"  # or "least_loaded", "random"
```

### 3. Hierarchical Pattern

The hierarchical pattern manages parent-child relationships with context propagation.

#### Use Cases
- **Risk Hierarchies**: Portfolio → Sub-portfolio → Position level risk
- **Regime-Based Systems**: Regime detector → Risk profiles → Strategies
- **Organizational Structures**: Department → Team → Individual traders

#### Implementation

```python
class HierarchicalAdapter:
    """Parent-child communication with context"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.parent = config['parent']
        self.children = config['children']
        self.bidirectional = config.get('bidirectional', True)
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure hierarchical connections"""
        parent = containers[self.parent]
        child_containers = [
            containers[child['name']] for child in self.children
        ]
        
        # Parent to children
        parent.event_bus.subscribe(
            EventType.CONTEXT,
            lambda event: self.propagate_context(event, child_containers)
        )
        
        if self.bidirectional:
            # Children to parent aggregation
            for child_config, child in zip(self.children, child_containers):
                weight = child_config.get('weight', 1.0)
                child.event_bus.subscribe(
                    EventType.RESULT,
                    lambda event, w=weight: self.aggregate_to_parent(event, parent, w)
                )
    
    def propagate_context(self, event: Event, children: List[Container]) -> None:
        """Propagate context from parent to children"""
        for child in children:
            # Enrich event with hierarchical context
            context_event = self.enrich_with_context(event, child)
            child.receive_event(context_event)
    
    def aggregate_to_parent(self, event: Event, parent: Container, weight: float) -> None:
        """Aggregate child results to parent"""
        # Apply weighting
        if hasattr(event, 'value') and isinstance(event.value, (int, float)):
            event.value *= weight
        parent.receive_event(event)
    
    def enrich_with_context(self, event: Event, child: Container) -> Event:
        """Add hierarchical context to event"""
        enriched = copy.deepcopy(event)
        enriched.hierarchy_level = child.level
        enriched.parent_context = self.parent
        return enriched
```

#### Configuration Example

```yaml
adapters:
  - type: "hierarchical"
    name: "regime_hierarchy"
    parent: "regime_classifier"
    children:
      - name: "conservative_risk_profile"
        weight: 0.3
        context_mapping:
          regime: "risk_level"
      - name: "balanced_risk_profile"
        weight: 0.5
        context_mapping:
          regime: "risk_level"
      - name: "aggressive_risk_profile"
        weight: 0.2
        context_mapping:
          regime: "risk_level"
    bidirectional: true
    context_propagation:
      - field: "regime"
        enrichment: "add_regime_parameters"
```

#### Variations

**Multi-Level Hierarchy**: Support deep hierarchies
```yaml
adapters:
  - type: "multi_level_hierarchy"
    name: "organizational_structure"
    levels:
      - parent: "portfolio_manager"
        children: ["equity_desk", "fixed_income_desk", "fx_desk"]
      - parent: "equity_desk"
        children: ["tech_sector", "energy_sector", "finance_sector"]
      - parent: "tech_sector"
        children: ["aapl_trader", "msft_trader", "googl_trader"]
```

### 4. Selective Pattern

The selective pattern routes events based on content and rules.

#### Use Cases
- **Signal Routing**: Route based on confidence, regime, or asset class
- **Order Routing**: Smart order routing based on order characteristics
- **Alert Routing**: Send alerts to appropriate handlers

#### Implementation

```python
class SelectiveAdapter:
    """Content-based routing"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.source = config['source']
        self.rules = config['rules']
        self.default_target = config.get('default_target')
        self.routing_rules = []
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure selective routing"""
        source = containers[self.source]
        
        # Parse and compile rules
        self.routing_rules = []
        for rule in self.rules:
            condition = self.compile_condition(rule['condition'])
            target = containers[rule['target']]
            priority = rule.get('priority', 0)
            self.routing_rules.append((condition, target, priority))
        
        # Sort by priority
        self.routing_rules.sort(key=lambda x: x[2], reverse=True)
        
        # Subscribe to source
        source.event_bus.subscribe(EventType.OUTPUT, self.route_selectively)
    
    def route_selectively(self, event: Event) -> None:
        """Route based on rules"""
        for condition, target, _ in self.routing_rules:
            if condition(event):
                target.receive_event(event)
                if not self.config.get('multi_match', False):
                    return  # First match only
        
        # Default routing
        if self.default_target:
            self.default_target.receive_event(event)
    
    def compile_condition(self, condition_str: str) -> Callable:
        """Compile condition string to callable"""
        # Safe evaluation of conditions
        namespace = {
            'event': None,
            're': re,
            'datetime': datetime
        }
        
        def evaluator(event):
            namespace['event'] = event
            return eval(condition_str, {"__builtins__": {}}, namespace)
        
        return evaluator
```

#### Configuration Example

```yaml
adapters:
  - type: "selective"
    name: "smart_signal_router"
    source: "signal_aggregator"
    rules:
      # High confidence signals to aggressive execution
      - condition: "event.confidence > 0.8 and event.signal_strength > 0.7"
        target: "aggressive_executor"
        priority: 10
        
      # Regime-specific routing
      - condition: "event.regime == 'TRENDING' and event.asset_class == 'EQUITY'"
        target: "trend_following_executor"
        priority: 8
        
      # Time-based routing
      - condition: "datetime.now().hour < 10"
        target: "opening_auction_executor"
        priority: 7
        
      # Risk-based routing
      - condition: "event.risk_score > 0.6"
        target: "risk_reduction_executor"
        priority: 9
        
    default_target: "standard_executor"
    multi_match: false  # Stop at first match
```

#### Variations

**ML-Based Routing**: Use ML models for routing decisions
```python
class MLSelectiveAdapter:
    """Machine learning based routing"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.model = self.load_routing_model(config['model_path'])
        self.containers = {}
        
    def route_selectively(self, event: Event) -> None:
        """Use ML model for routing decision"""
        features = self.extract_features(event)
        prediction = self.model.predict(features)
        target_name = self.model.classes_[prediction[0]]
        target = self.containers[target_name]
        target.receive_event(event)
    
    def load_routing_model(self, path: str):
        """Load the ML routing model"""
        # Implementation to load model
        pass
    
    def extract_features(self, event: Event):
        """Extract features from event for ML model"""
        # Implementation to extract features
        pass
```

### 5. Composite Pattern

The composite pattern combines multiple adapters for complex flows.

#### Use Cases
- **Multi-Stage Workflows**: Combine different patterns at different stages
- **Hybrid Systems**: Mix patterns based on requirements
- **Dynamic Flows**: Change patterns based on runtime conditions

#### Implementation

```python
class CompositeAdapter:
    """Combines multiple adapter patterns"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.sub_adapters = []
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Setup all sub-adapters"""
        factory = AdapterFactory()
        
        for adapter_config in self.config['adapters']:
            sub_adapter = factory.create(adapter_config)
            sub_adapter.setup(containers)
            self.sub_adapters.append(sub_adapter)
            
        # Connect sub-adapters if needed
        if self.config.get('connect_adapters', False):
            self.connect_sub_adapters()
    
    def start(self) -> None:
        """Start all sub-adapters"""
        for adapter in self.sub_adapters:
            adapter.start()
    
    def stop(self) -> None:
        """Stop all sub-adapters"""
        for adapter in self.sub_adapters:
            adapter.stop()
    
    def connect_sub_adapters(self) -> None:
        """Connect sub-adapters for complex flows"""
        # Implementation to connect adapters
        pass
```

#### Configuration Example

```yaml
adapters:
  - type: "composite"
    name: "complex_trading_flow"
    adapters:
      # Stage 1: Broadcast market data
      - type: "broadcast"
        source: "market_data"
        targets: ["strategy_a", "strategy_b", "strategy_c"]
        
      # Stage 2: Selective routing of signals
      - type: "selective"
        source: "signal_aggregator"
        rules:
          - condition: "event.strategy == 'strategy_a'"
            target: "risk_manager_a"
          - condition: "event.strategy == 'strategy_b'"
            target: "risk_manager_b"
            
      # Stage 3: Pipeline to execution
      - type: "pipeline"
        containers: ["risk_manager_a", "position_sizer", "executor"]
        
    connect_adapters: true
```

## Advanced Patterns

### 1. Adapter Composition

Adapters can be composed to create more sophisticated communication patterns. This is particularly powerful when you need to combine the characteristics of multiple patterns.

#### Selective Broadcast Example

A common composition is the "Selective Broadcast" - broadcasting to multiple targets but with selective filtering per target:

```python
class SelectiveBroadcastAdapter:
    """Broadcasts events with per-target filtering"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.source = config['source']
        self.targets = config['targets']  # List of {target, filter} objects
        
    def setup(self, containers: Dict[str, Container]) -> None:
        """Setup selective broadcast"""
        source_container = containers[self.source]
        
        # Create filtered paths for each target
        for target_config in self.targets:
            target_name = target_config['name']
            target_filter = target_config.get('filter')
            target_container = containers[target_name]
            
            # Subscribe with filtering
            source_container.event_bus.subscribe(
                EventType.OUTPUT,
                lambda event, t=target_container, f=target_filter: 
                    self._filtered_forward(event, t, f)
            )
    
    def _filtered_forward(self, event: Event, target: Container, filter_fn: Optional[Callable]):
        """Forward event only if it passes the filter"""
        if filter_fn is None or filter_fn(event):
            target.receive_event(event)
```

#### Configuration Example

```yaml
# Selective broadcast - different strategies get different signals
adapters:
  - type: "selective_broadcast"
    name: "smart_signal_distribution"
    source: "signal_generator"
    targets:
      # High-frequency strategy only gets strong, short-term signals
      - name: "high_freq_strategy"
        filter: "lambda e: e.strength > 0.8 and e.timeframe == '1m'"
        
      # Swing trader gets medium-term signals
      - name: "swing_strategy"
        filter: "lambda e: e.timeframe in ['1h', '4h'] and e.confidence > 0.6"
        
      # Long-term investor gets daily signals with high confidence
      - name: "investor_strategy"
        filter: "lambda e: e.timeframe == '1d' and e.confidence > 0.7"
        
      # Risk monitor gets ALL signals for oversight
      - name: "risk_monitor"
        # No filter - receives everything
```

#### Dynamic Composition

Compose adapters dynamically at runtime:

```python
class AdapterComposer:
    """Dynamically compose adapters"""
    
    def compose(self, *adapters: CommunicationAdapter) -> CompositeAdapter:
        """Compose multiple adapters into one"""
        config = {
            'type': 'composite',
            'name': f"composed_{uuid.uuid4().hex[:8]}",
            'adapters': [a.config for a in adapters]
        }
        
        composite = CompositeAdapter(config['name'], config)
        composite.sub_adapters = list(adapters)
        return composite
    
    def create_selective_broadcast(self, broadcast_config, selective_config):
        """Create a selective broadcast from broadcast + selective patterns"""
        # First broadcast to intermediate containers
        broadcast = BroadcastAdapter("broadcast_stage", broadcast_config)
        
        # Then apply selective routing
        selective = SelectiveAdapter("selective_stage", selective_config)
        
        # Compose them
        return self.compose(broadcast, selective)
    
    def chain(self, *adapters: CommunicationAdapter) -> PipelineAdapter:
        """Chain adapters in sequence"""
        # Extract container sequence
        containers = []
        for adapter in adapters:
            if hasattr(adapter, 'containers'):
                containers.extend(adapter.containers)
                
        config = {
            'type': 'pipeline',
            'name': f"chained_{uuid.uuid4().hex[:8]}",
            'containers': containers
        }
        
        return PipelineAdapter(config['name'], config)
```

Usage:
```python
# Compose adapters
composer = AdapterComposer()

# Example 1: Create complex flow
data_distribution = BroadcastAdapter("data_dist", {...})
risk_routing = SelectiveAdapter("risk_route", {...})
execution_pipeline = PipelineAdapter("exec_pipe", {...})

complex_flow = composer.compose(
    data_distribution,
    risk_routing,
    execution_pipeline
)

# Example 2: Create selective broadcast
selective_broadcast = composer.create_selective_broadcast(
    broadcast_config={'source': 'signals', 'targets': ['s1', 's2', 's3']},
    selective_config={'rules': [...]}  # Apply filtering after broadcast
)
```

#### Real-World Composition Examples

**1. Tiered Signal Distribution**
```yaml
# Compose broadcast + selective for sophisticated routing
adapters:
  - type: "composite"
    name: "tiered_distribution"
    adapters:
      # First: broadcast to signal processors
      - type: "broadcast"
        source: "raw_signals"
        targets: ["enhancer_1", "enhancer_2", "validator"]
        
      # Then: selective routing based on enhanced signals
      - type: "selective"
        source: "signal_aggregator"
        rules:
          - condition: "signal.quality == 'premium'"
            target: "premium_strategies"
          - condition: "signal.quality == 'standard'"
            target: "standard_strategies"
```

**2. Fault-Tolerant Pipeline**
```yaml
# Compose pipeline + broadcast for redundancy
adapters:
  - type: "composite"
    name: "fault_tolerant_flow"
    adapters:
      # Main pipeline
      - type: "pipeline"
        containers: ["data", "strategy", "primary_executor"]
        
      # Broadcast critical events to backup systems
      - type: "broadcast"
        source: "strategy"
        targets: ["backup_executor", "risk_monitor", "audit_log"]
```

### 2. Dynamic Adapter Reconfiguration

Modify adapter behavior at runtime:

```python
class DynamicAdapter:
    """Adapter with runtime reconfiguration"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._config_version = 0
        self._config_lock = threading.RLock()
        
    def reconfigure(self, new_config: Dict[str, Any]) -> None:
        """Reconfigure adapter at runtime"""
        with self._config_lock:
            # Validate new configuration
            self.validate_config(new_config)
            
            # Store old config
            old_config = self.config.copy()
            
            try:
                # Apply new configuration
                self.config.update(new_config)
                self._config_version += 1
                
                # Reestablish connections
                self.apply_configuration()
                
                print(f"Reconfigured adapter {self.name} to version {self._config_version}")
                
            except Exception as e:
                # Rollback on failure
                self.config = old_config
                print(f"Reconfiguration failed: {e}")
                raise
    
    def apply_configuration(self) -> None:
        """Apply configuration changes"""
        # Re-setup connections based on new config
        self.stop()
        self.setup(self.containers)
        self.start()
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration"""
        # Implementation to validate config
        pass
```

Usage:
```python
# Create dynamic adapter
adapter = DynamicAdapter("dynamic_router", {
    'type': 'selective',
    'rules': [
        {'condition': 'event.urgency == "high"', 'target': 'fast_executor'}
    ]
})

# Reconfigure at runtime
adapter.reconfigure({
    'rules': [
        {'condition': 'event.urgency == "critical"', 'target': 'ultra_fast_executor'},
        {'condition': 'event.urgency == "high"', 'target': 'fast_executor'},
        {'condition': 'event.urgency == "normal"', 'target': 'standard_executor'}
    ]
})
```

### 3. Performance Metrics Pattern

Track detailed performance metrics:

```python
class MetricsAdapter:
    """Adapter with comprehensive metrics"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.metrics = {
            'events_routed': Counter(),
            'routing_latency': Histogram(buckets=[0.001, 0.005, 0.01, 0.05, 0.1]),
            'errors': Counter(),
            'active_routes': Gauge()
        }
        
    def route_event(self, event: Event, source: Container) -> None:
        """Route with metrics tracking"""
        start = time.perf_counter()
        
        try:
            # Actual routing logic
            self._perform_routing(event, source)
            
            # Track success metrics
            self.metrics['events_routed'].increment({
                'source': source.name,
                'event_type': event.type
            })
            
        except Exception as e:
            # Track error metrics
            self.metrics['errors'].increment({
                'source': source.name,
                'error_type': type(e).__name__
            })
            raise
            
        finally:
            # Track latency
            latency = time.perf_counter() - start
            self.metrics['routing_latency'].observe(latency, {
                'adapter': self.name
            })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            'total_events': self.metrics['events_routed'].value(),
            'error_rate': self.metrics['errors'].value() / max(1, self.metrics['events_routed'].value()),
            'p99_latency': self.metrics['routing_latency'].quantile(0.99),
            'p95_latency': self.metrics['routing_latency'].quantile(0.95),
            'p50_latency': self.metrics['routing_latency'].quantile(0.50)
        }
    
    def _perform_routing(self, event: Event, source: Container) -> None:
        """Perform actual routing - override in subclasses"""
        pass
```

## Type Flow Analysis with Adapter Patterns

Each adapter pattern has specific implications for type flow analysis. Understanding how event types flow through different patterns helps ensure correct system configuration.

### Pipeline Pattern Type Flow

The pipeline pattern creates a linear transformation of event types:

```python
class PipelineTypeAnalyzer:
    """Analyze type flow through pipeline adapters"""
    
    def analyze_pipeline(self, pipeline_config: Dict[str, Any], 
                        type_analyzer: TypeFlowAnalyzer) -> List[TypeTransition]:
        """Analyze type transitions in a pipeline"""
        containers = pipeline_config['containers']
        transitions = []
        
        for i in range(len(containers) - 1):
            source = containers[i]
            target = containers[i + 1]
            
            # Get type transformations
            source_outputs = type_analyzer.get_container_outputs(source)
            target_inputs = type_analyzer.get_container_inputs(target)
            
            # Check compatibility
            compatible_types = source_outputs & target_inputs
            if not compatible_types:
                transitions.append(TypeTransition(
                    source=source,
                    target=target,
                    status="INCOMPATIBLE",
                    error=f"No compatible types between {source} and {target}"
                ))
            else:
                transitions.append(TypeTransition(
                    source=source,
                    target=target,
                    status="OK",
                    types=compatible_types
                ))
        
        return transitions
```

### Broadcast Pattern Type Flow

Broadcast patterns must ensure all targets can handle the source's event types:

```python
class BroadcastTypeAnalyzer:
    """Analyze type flow through broadcast adapters"""
    
    def analyze_broadcast(self, broadcast_config: Dict[str, Any],
                         type_analyzer: TypeFlowAnalyzer) -> BroadcastAnalysis:
        """Analyze broadcast type compatibility"""
        source = broadcast_config['source']
        targets = broadcast_config['targets']
        
        source_types = type_analyzer.get_container_outputs(source)
        analysis = BroadcastAnalysis(source=source, source_types=source_types)
        
        for target in targets:
            target_types = type_analyzer.get_container_inputs(target)
            compatible = source_types & target_types
            
            if not compatible:
                analysis.add_incompatible_target(target, 
                    f"Cannot receive any events from {source}")
            else:
                analysis.add_compatible_target(target, compatible)
        
        return analysis
```

### Hierarchical Pattern Type Flow

Hierarchical patterns involve bidirectional type flow:

```python
class HierarchicalTypeAnalyzer:
    """Analyze type flow in hierarchical adapters"""
    
    def analyze_hierarchy(self, hierarchy_config: Dict[str, Any],
                         type_analyzer: TypeFlowAnalyzer) -> HierarchyAnalysis:
        """Analyze parent-child type flows"""
        parent = hierarchy_config['parent']
        children = hierarchy_config['children']
        
        # Downward flow (parent to children)
        parent_context_types = type_analyzer.get_context_events(parent)
        
        # Upward flow (children to parent)
        child_result_types = set()
        for child in children:
            child_name = child['name'] if isinstance(child, dict) else child
            child_result_types.update(
                type_analyzer.get_result_events(child_name)
            )
        
        return HierarchyAnalysis(
            parent=parent,
            children=children,
            downward_types=parent_context_types,
            upward_types=child_result_types
        )
```

### Selective Pattern Type Flow

Selective patterns route specific event types based on rules:

```python
class SelectiveTypeAnalyzer:
    """Analyze type flow through selective adapters"""
    
    def analyze_selective(self, selective_config: Dict[str, Any],
                         type_analyzer: TypeFlowAnalyzer) -> SelectiveAnalysis:
        """Analyze selective routing type coverage"""
        source = selective_config['source']
        rules = selective_config['rules']
        
        source_types = type_analyzer.get_container_outputs(source)
        analysis = SelectiveAnalysis(source=source)
        
        # Check which event types are covered by rules
        covered_types = set()
        
        for rule in rules:
            # Analyze rule to determine which event types it handles
            handled_types = self._analyze_rule_coverage(rule, source_types)
            covered_types.update(handled_types)
            
            target = rule['target']
            target_inputs = type_analyzer.get_container_inputs(target)
            
            # Check compatibility
            compatible = handled_types & target_inputs
            if not compatible:
                analysis.add_warning(
                    f"Rule {rule['condition']} routes to {target} "
                    f"but no compatible types"
                )
        
        # Check for uncovered types
        uncovered = source_types - covered_types
        if uncovered and not selective_config.get('default_target'):
            analysis.add_error(
                f"Event types {uncovered} have no routing rules "
                f"and no default target"
            )
        
        return analysis
```

### Composite Pattern Type Flow

Composite patterns require analyzing the combined flow:

```python
class CompositeTypeAnalyzer:
    """Analyze type flow through composite adapters"""
    
    def analyze_composite(self, composite_config: Dict[str, Any],
                         type_analyzer: TypeFlowAnalyzer) -> CompositeAnalysis:
        """Analyze complex multi-adapter flows"""
        sub_adapters = composite_config['adapters']
        analysis = CompositeAnalysis()
        
        # Build combined flow graph
        flow_graph = self._build_composite_graph(sub_adapters)
        
        # Analyze end-to-end paths
        for source in flow_graph.sources:
            for sink in flow_graph.sinks:
                paths = flow_graph.find_paths(source, sink)
                
                for path in paths:
                    path_analysis = self._analyze_path_types(path, type_analyzer)
                    analysis.add_path_analysis(path_analysis)
        
        return analysis
```

### Visualizing Adapter Type Flows

Visualize how types flow through adapter configurations:

```python
class AdapterTypeFlowVisualizer:
    """Visualize type flow through adapter patterns"""
    
    def visualize_adapter_flows(self, adapters: List[Dict[str, Any]],
                               type_analyzer: TypeFlowAnalyzer) -> str:
        """Generate visualization of adapter type flows"""
        lines = ["Adapter Type Flow Analysis", "=" * 50, ""]
        
        for adapter in adapters:
            adapter_type = adapter['type']
            adapter_name = adapter.get('name', adapter_type)
            
            lines.append(f"Adapter: {adapter_name} ({adapter_type})")
            lines.append("-" * 30)
            
            if adapter_type == 'pipeline':
                analysis = PipelineTypeAnalyzer().analyze_pipeline(
                    adapter, type_analyzer
                )
                lines.extend(self._format_pipeline_analysis(analysis))
                
            elif adapter_type == 'broadcast':
                analysis = BroadcastTypeAnalyzer().analyze_broadcast(
                    adapter, type_analyzer
                )
                lines.extend(self._format_broadcast_analysis(analysis))
                
            # ... other adapter types
            
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_mermaid_adapter_diagram(self, adapters: List[Dict[str, Any]],
                                       type_analyzer: TypeFlowAnalyzer) -> str:
        """Generate Mermaid diagram showing adapter patterns and types"""
        lines = ["graph LR"]
        
        # Add adapter pattern visualizations
        for i, adapter in enumerate(adapters):
            adapter_type = adapter['type']
            
            if adapter_type == 'pipeline':
                # Show linear flow with types
                containers = adapter['containers']
                for j in range(len(containers) - 1):
                    source = containers[j]
                    target = containers[j + 1]
                    types = self._get_flowing_types(source, target, type_analyzer)
                    
                    lines.append(
                        f"    {source} -->|{self._format_types(types)}| {target}"
                    )
                    
            elif adapter_type == 'broadcast':
                # Show fanout with types
                source = adapter['source']
                for target in adapter['targets']:
                    types = self._get_flowing_types(source, target, type_analyzer)
                    lines.append(
                        f"    {source} ==>|{self._format_types(types)}| {target}"
                    )
        
        return "\n".join(lines)
```

### Integration Example

Combining adapter patterns with type flow analysis:

```python
# Create type flow analyzer
type_analyzer = TypeFlowAnalyzer()

# Define adapter configuration
adapters = [
    {
        'type': 'pipeline',
        'name': 'main_flow',
        'containers': ['data_source', 'indicators', 'strategy', 'risk', 'execution']
    },
    {
        'type': 'broadcast',
        'name': 'signal_distribution',
        'source': 'strategy',
        'targets': ['monitor', 'logger', 'analyzer']
    }
]

# Analyze type flows
for adapter in adapters:
    if adapter['type'] == 'pipeline':
        analysis = PipelineTypeAnalyzer().analyze_pipeline(adapter, type_analyzer)
        if any(t.status == "INCOMPATIBLE" for t in analysis):
            raise ConfigurationError(f"Type incompatibility in {adapter['name']}")

# Visualize the complete type flow
visualizer = AdapterTypeFlowVisualizer()
print(visualizer.visualize_adapter_flows(adapters, type_analyzer))

# Generate diagram
with open('adapter_type_flow.mmd', 'w') as f:
    f.write(visualizer.generate_mermaid_adapter_diagram(adapters, type_analyzer))
```

This integration ensures that:
- Adapter patterns are type-safe
- Configuration errors are caught early
- Event flows are validated before runtime
- Complex adapter compositions maintain type correctness

## Pattern Selection Guide

Choose the right pattern based on your requirements:

| Pattern | Use When | Performance | Complexity |
|---------|----------|-------------|------------|
| Pipeline | Sequential processing needed | High | Low |
| Broadcast | One-to-many distribution | Medium | Low |
| Hierarchical | Parent-child relationships | Medium | Medium |
| Selective | Content-based routing | Medium | Medium |
| Composite | Complex flows needed | Variable | High |

### Decision Tree

```
Start
│
├─ Need sequential processing?
│  └─ Yes → Pipeline Pattern
│
├─ Need one-to-many distribution?
│  └─ Yes → Broadcast Pattern
│
├─ Have parent-child relationships?
│  └─ Yes → Hierarchical Pattern
│
├─ Need content-based routing?
│  └─ Yes → Selective Pattern
│
└─ Need combination of patterns?
   └─ Yes → Composite Pattern
```

## Best Practices

### 1. Pattern Guidelines

- **Start Simple**: Begin with basic patterns, evolve as needed
- **Single Responsibility**: Each adapter should have one clear purpose
- **Configuration Over Code**: Maximize configurability
- **Performance First**: Choose patterns that match performance needs

### 2. Common Anti-Patterns

**Anti-Pattern: Adapter Spaghetti**
```yaml
# BAD: Too many interconnected adapters
adapters:
  - type: "broadcast"
    source: "a"
    targets: ["b", "c", "d", "e", "f", "g"]
  - type: "selective"
    source: "b"
    rules: [...]  # 20+ rules
  - type: "broadcast"
    source: "c"
    targets: ["d", "e", "f"]
  # ... 10 more adapters
```

**Better: Organized Hierarchy**
```yaml
# GOOD: Clear, organized structure
adapters:
  - type: "composite"
    name: "main_flow"
    adapters:
      - type: "broadcast"
        name: "data_distribution"
        source: "data"
        targets: ["strategies"]
      - type: "pipeline"
        name: "execution_flow"
        containers: ["strategies", "risk", "execution"]
```

**Anti-Pattern: Over-Engineering**
```python
# BAD: Complex adapter for simple needs
class QuantumEntangledBlockchainAdapter:
    # 500 lines of complex code
    pass
```

**Better: Simple and Clear**
```python
# GOOD: Simple adapter that does the job
class SimpleRoutingAdapter:
    # 50 lines of clear code
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        
    def route(self, event: Event, target: Container):
        target.receive_event(event)
```

### 3. Testing Patterns

Test each pattern thoroughly:

```python
class PatternTestBase:
    """Base functionality for pattern tests"""
    
    def test_basic_routing(self, adapter, source, target):
        """Test basic event routing"""
        event = TestEvent()
        adapter.handle_event(event, source)
        assert target.received_events[-1] == event
        
    def test_error_handling(self, adapter, source):
        """Test error handling"""
        # Force an error
        event = ErrorEvent()
        with pytest.raises(AdapterError):
            adapter.handle_event(event, source)
            
    def test_performance(self, adapter, source, target):
        """Test performance characteristics"""
        events = [TestEvent() for _ in range(1000)]
        
        start = time.perf_counter()
        for event in events:
            adapter.handle_event(event, source)
        duration = time.perf_counter() - start
        
        assert duration < 0.1  # 100ms for 1000 events
```

## Conclusion

Adapter patterns provide flexible, reusable solutions for container communication. By understanding and applying these patterns appropriately, you can build maintainable, scalable trading systems that adapt to changing requirements without code modifications.

Key takeaways:
1. **Choose the Right Pattern**: Match pattern to use case
2. **Keep It Simple**: Don't over-engineer
3. **Configure, Don't Code**: Maximize configuration flexibility
4. **Monitor Performance**: Track metrics for all adapters
5. **Test Thoroughly**: Ensure patterns work as expected

The true power of adapter patterns lies in their composability - simple patterns can be combined to create sophisticated communication flows while maintaining clarity and maintainability.
