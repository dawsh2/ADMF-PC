# ADMF-PC User Guide & Architecture Overview

## Core Architectural Principles

ADMF-PC is built on a clean, event-driven architecture with these key principles:

1. **Event-Driven Communication**: All components communicate through events
2. **Container Isolation**: Each container has its own event bus for true isolation
3. **Parent-Child Hierarchy**: Events flow naturally up/down the container tree
4. **Stateless Strategies**: Trading logic is pure functions, state lives in containers
5. **Configuration-Driven**: Everything is wired through YAML, no hardcoded dependencies

## The Event System

### Event Bus Architecture

Each container gets its own isolated `EventBus` with these capabilities:

```python
# Local publishing (within container)
container.event_bus.publish(event)

# Publishing to parent (propagates up the tree)
container.publish_event(event, target_scope="parent")

# Parent automatically forwards to all children
# This enables natural event flow without explicit routing
```

### Event Flow Patterns

```
Root Container
├── Data Container → publishes BAR events to parent
├── Feature Container → subscribes to BARs from parent, publishes FEATURES to parent  
├── Strategy Container → subscribes to BARs/FEATURES from parent, publishes SIGNALs to parent
├── Portfolio Container → subscribes to SIGNALs from parent (filtered by strategy_id)
└── Execution Container → subscribes to ORDERs from parent, publishes FILLs to parent
```

### Event Filtering

The event bus supports filtered subscriptions - critical for multi-strategy systems:

```python
# Portfolio subscribes only to signals from its assigned strategies
bus.subscribe(
    EventType.SIGNAL.value,
    portfolio.handle_signal,
    filter_func=lambda e: e.payload.get('strategy_id') in ['momentum_1', 'pairs_2']
)
```

## Container System

### Container Hierarchy

Containers form a tree structure with automatic event propagation:

1. **Root Container**: Owns the root event bus, all events flow through here
2. **Child Containers**: Specialized containers for data, features, strategies, etc.
3. **Parent-Child Communication**: 
   - Children publish to parent with `target_scope="parent"`
   - Parent forwards events to all children automatically
   - No explicit routing needed!

### Container Lifecycle

```python
# Containers are created dynamically from config
container = Container(config)
container.add_component('feature_pipeline', FeaturePipeline())
container.initialize()
container.start()
container.execute()  # Begins event-driven processing
container.stop()
container.cleanup()
```

## Event Tracing & Metrics

### Hierarchical Event Storage

Events are the sole source of truth for metrics:

```
workspaces/
└── workflow_id/
    ├── root_container_id/
    │   └── events.jsonl
    ├── portfolio_container_id/
    │   └── events.jsonl
    └── strategy_container_id/
        └── events.jsonl
```

### Memory-Efficient Tracing

Event traces are pruned based on retention policies:

1. **Trade-Complete**: Keep events only for open trades, prune on close
2. **Rolling Window**: Keep last N events
3. **Time-Based**: Keep events from last T minutes
4. **Sparse**: Only store important event types (SIGNAL, FILL, etc.)

### Metrics from Events

MetricsObserver attaches to event bus and computes metrics incrementally:
- No separate metrics storage
- Metrics derived from event stream
- Memory-efficient incremental calculation

## Dynamic Configuration & Topology

### Topology Definition (topology.py)

Topologies define container structures and relationships:

```yaml
containers:
  - name: root
    type: root
    containers:  # Children defined inline
      - name: data_streamer
        type: data
        config:
          symbols: [SPY, QQQ]
          
      - name: feature_engine  
        type: features
        config:
          components: ['feature_pipeline']
          
      - name: momentum_strategy
        type: strategy
        config:
          components: ['strategy_executor']
          strategy_function: 'momentum_strategy'
          features_required: ['sma_20', 'rsi_14']
```

### Dynamic Instantiation

The topology builder:
1. Creates containers from config
2. Establishes parent-child relationships
3. Injects required components
4. **Components come from configuration, not factory!**

### Feature Inference

When a strategy declares required features:
```yaml
strategy_function: momentum_strategy
features_required: ['sma_20', 'rsi_14']
```

The topology builder:
1. Inspects strategy requirements
2. Configures feature pipeline with needed features
3. No manual wiring needed!

## Workflow Orchestration

### Sequencer

Manages execution phases:
1. **Initialize Phase**: Create containers, establish relationships
2. **Execution Phase**: Start event flow, let containers react
3. **Cleanup Phase**: Collect results, save traces

### Coordinator

Orchestrates multi-phase workflows:
- Walk-forward optimization
- Parameter sweeps  
- Monte Carlo simulations
- All through configuration!

## Clean Architecture Guidelines

### DO:
- Publish events to parent scope for propagation
- Use stateless strategy functions
- Configure everything through YAML
- Let parent-child hierarchy handle routing
- Use event filtering for multi-strategy isolation

### DON'T:
- Create wrapper components unnecessarily
- Add mock components to factories
- Hardcode component relationships
- Try to manually route between siblings
- Store state in strategies

## Current State & TODOs

### ✅ Working:
- Event bus with parent-child propagation
- Container hierarchy creation
- Event tracing with hierarchical storage
- BAR event generation from data containers

### 🔧 Issues to Fix:

1. **Factory Cleanup** (Priority 1)
   - Remove mock component creation
   - Remove strategy_wrapper from factory
   - Factory should only map names to real components
   - Component selection should be configuration-driven

2. **Data Handler Fix** (Priority 2)
   - Change data handler to publish to parent scope
   - Currently publishes only to local bus
   - Should be: `self.container.publish_event(event, target_scope="parent")`

3. **Feature Container** (Priority 3)
   - Create proper FeatureContainer component that:
     - Subscribes to BAR events from parent bus
     - Maintains FeatureHub for incremental calculation
     - Calls configured strategy functions with features
     - Publishes SIGNAL events to parent
   - Remove StrategyWrapper - this logic belongs in FeatureContainer

### 🎯 Next Steps:

1. Clean up factory.py - remove all mocks and wrappers
2. Fix data handler to publish to parent 
3. Create clean FeatureContainer implementation
4. Validate event flow: Data → Root → Features → Root → Portfolio

### 📝 Key Insights to Preserve:

- The parent-child event propagation eliminates need for complex routing
- Event filtering at subscription time prevents wrong signals reaching portfolios  
- All metrics come from events - no separate metrics system needed
- Configuration drives everything - no hardcoded dependencies
- Stateless strategies enable perfect parallelization

## Example: Proper Event Flow

```python
# Data container publishes BAR
data_container.publish_event(bar_event, target_scope="parent")  # → goes to root

# Root container receives and forwards to all children
# Feature container (child of root) receives BAR automatically

# Feature container processes and publishes 
feature_container.publish_event(signal_event, target_scope="parent")  # → goes to root

# Root forwards to all children
# Portfolio container (child of root) receives if filter matches
```

This is the clean architecture - no routers, no complex wiring, just natural parent-child event flow!