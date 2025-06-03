# Adapter Architecture: Practical Communication Patterns

## Overview

The ADMF-PC adapter architecture solves a practical problem: how to enable flexible communication between containers without hardcoding routing logic. Rather than embedding communication patterns in container code, adapters externalize this logic into configurable components.

## The Real Problem

When you have a trading system with multiple strategies, classifiers, and risk managers, the communication patterns can vary significantly based on how you organize your combinatorial search:

```python
# Problem: Communication logic mixed with business logic
class TradingContainer:
    def _route_events(self, event):
        # This gets messy fast
        if self.search_mode == "strategy_first":
            # Route to all classifiers for this strategy
            for classifier in self.child_classifiers:
                classifier.receive(event)
        elif self.search_mode == "classifier_first":
            # Route to all strategies for this classifier
            for strategy in self.child_strategies:
                strategy.receive(event)
        # More conditions accumulate...
```

## The Adapter Solution

Adapters separate communication patterns from container logic:

```python
# Clean separation
class Container:
    def process(self, event):
        # Pure business logic only
        result = self.calculate(event)
        self.publish(result)  # Adapter handles routing

# Configuration determines routing
adapters:
  - type: "broadcast"    # One-to-many for data distribution
  - type: "pipeline"     # Sequential for execution flow
  - type: "selective"    # Conditional for risk-based routing
```

## Core Adapter Patterns

### 1. Pipeline Pattern - Linear Processing Flow

The fundamental pattern for the core trading flow:

```
Data → Indicators → Strategy → Risk → Execution → Portfolio
```

**Use when**: You have sequential processing stages
**Example**: Basic backtest flow

```yaml
adapters:
  - type: "pipeline"
    containers: ["data", "indicators", "strategy", "risk", "execution"]
```

### 2. Broadcast Pattern - Shared Data Distribution

Essential for efficient combinatorial search:

```
                 ┌→ Strategy A
Indicators Hub ──├→ Strategy B  
                 └→ Strategy C
```

**Use when**: Multiple consumers need the same data
**Example**: All strategies receiving the same market data

```yaml
adapters:
  - type: "broadcast"
    source: "indicator_hub"
    targets: ["momentum_strategy", "mean_reversion", "breakout"]
```

### 3. Hierarchical Pattern - Parent-Child Relationships

Perfect for nested combinatorial search:

```
Classifier
  ├─ Risk Profile A
  │   └─ Strategies...
  └─ Risk Profile B
      └─ Strategies...
```

**Use when**: You have nested container hierarchies
**Example**: Regime-based strategy selection

```yaml
adapters:
  - type: "hierarchical"
    parent: "hmm_classifier"
    children: ["conservative_risk", "aggressive_risk"]
    propagate_context: true
```

### 4. Selective Pattern - Conditional Routing

Handles dynamic routing based on event content:

```yaml
adapters:
  - type: "selective"
    source: "signal_generator"
    rules:
      - condition: "signal.confidence > 0.8"
        target: "aggressive_executor"
      - condition: "signal.confidence < 0.3"
        target: "paper_trader"
      - condition: "default"
        target: "normal_executor"
```

## Practical Benefits

### 1. Clean Separation of Concerns

**Without Adapters**:
```python
class Strategy:
    def process(self):
        # Business logic mixed with routing
        signal = self.calculate()
        if self.mode == "backtest":
            self.execution.receive(signal)
        elif self.mode == "optimization":
            self.logger.receive(signal)
        # Gets messy fast...
```

**With Adapters**:
```python
class Strategy:
    def process(self):
        # Only business logic
        signal = self.calculate()
        self.publish(signal)  # Routing handled by adapter
```

### 2. Flexible Combinatorial Search

Change search organization without changing code:

```yaml
# Search mode 1: Fix strategy, vary classifiers
adapters:
  - type: "broadcast"
    source: "momentum_strategy"
    targets: ["hmm_classifier", "pattern_classifier", "ml_classifier"]

# Search mode 2: Fix classifier, vary strategies  
adapters:
  - type: "broadcast"
    source: "hmm_classifier"
    targets: ["momentum", "mean_reversion", "breakout", "pairs"]
```

### 3. Performance Optimization

Adapters can batch events and optimize communication:

```yaml
adapters:
  - type: "broadcast"
    source: "data_stream"
    targets: ["strategy_1", "strategy_2", "strategy_3"]
    tier: "fast"
    batch_size: 1000  # Batch for efficiency
```

### 4. Easy Testing

Test communication patterns independently:

```python
def test_broadcast_adapter():
    adapter = BroadcastAdapter(source="test", targets=["a", "b", "c"])
    # Test that events reach all targets
    
def test_selective_adapter():
    adapter = SelectiveAdapter(rules=[...])
    # Test routing logic without real containers
```

## Real-World Example: Combinatorial Strategy Search

Here's how adapters enable efficient combinatorial search:

```yaml
# Searching for best strategy parameters across different market regimes
workflow: "combinatorial_search"

# Level 1: Broadcast market data to all classifiers
adapters:
  - name: "data_distribution"
    type: "broadcast"
    source: "market_data"
    targets: ["hmm_classifier", "volatility_classifier", "pattern_classifier"]

# Level 2: Each classifier tests all strategies
  - name: "hmm_to_strategies"
    type: "broadcast"
    source: "hmm_classifier"
    targets: ["momentum_variants", "reversion_variants"]
    
  - name: "volatility_to_strategies"
    type: "broadcast"
    source: "volatility_classifier"
    targets: ["momentum_variants", "reversion_variants"]

# Level 3: Route high-confidence signals to execution
  - name: "signal_routing"
    type: "selective"
    source: "signal_aggregator"
    rules:
      - condition: "signal.sharpe_ratio > 1.5"
        target: "live_executor"
      - condition: "signal.sharpe_ratio > 1.0"
        target: "paper_executor"
      - condition: "default"
        target: "logger_only"
```

## Integration with Container Hierarchies

Adapters naturally complement hierarchical container organization:

```python
# Containers handle business logic and hierarchy
classifier_container = Container("hmm_classifier")
risk_container = Container("conservative", parent=classifier_container)
strategy_container = Container("momentum", parent=risk_container)

# Adapters handle communication between levels
hierarchical_adapter = HierarchicalAdapter(
    parent="hmm_classifier",
    children=["conservative", "aggressive"],
    context_propagation=True  # Share regime context
)
```

## Best Practices

### 1. Match Adapter to Use Case

| Use Case | Adapter Type | Why |
|----------|--------------|-----|
| Sequential processing | Pipeline | Natural flow |
| Multiple consumers | Broadcast | Efficient sharing |
| Nested search | Hierarchical | Preserves hierarchy |
| Dynamic routing | Selective | Content-based decisions |

### 2. Keep Business Logic in Containers

Adapters should only handle routing, not processing:

```python
# Good: Adapter just routes
class PipelineAdapter:
    def route(self, event, source):
        next_container = self.get_next(source)
        next_container.receive(event)

# Bad: Adapter processes data
class PipelineAdapter:
    def route(self, event, source):
        modified_event = self.transform(event)  # Don't do this!
        next_container.receive(modified_event)
```

### 3. Use Configuration for Flexibility

Define all routing in configuration, not code:

```yaml
# Easy to modify without code changes
adapters:
  - type: "pipeline"
    containers: ["data", "strategy", "risk", "execution"]
    
  # Can add new routing without touching code
  - type: "broadcast"
    source: "risk_alerts"
    targets: ["logger", "dashboard", "email_notifier"]
```

## Summary

Adapters in ADMF-PC serve a practical purpose: they enable flexible communication patterns without coupling containers to specific routing logic. This is especially valuable for:

1. **Combinatorial search** - Easy to reorganize container hierarchies
2. **Testing** - Communication patterns can be tested independently
3. **Performance** - Batching and optimization without changing business logic
4. **Maintenance** - Routing changes don't require code modifications

The key insight: **Communication patterns are infrastructure, not business logic**. By separating them into adapters, we get flexibility without complexity.
