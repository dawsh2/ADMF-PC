# ADMF-PC Architecture: Practical Overview

## Core Insight: It's a Pipeline with Variations

At its heart, ADMF-PC implements the fundamental trading pipeline:

```
BAR → SIGNAL → ORDER → FILL → UPDATE
```

The architectural complexity exists to handle practical variations:
- Multiple strategies need the same data (broadcast)
- Different regimes need different strategies (hierarchical/selective)
- Components need to be tested in different combinations (invertible hierarchies)

## The Real Problems Being Solved

### 1. Combinatorial Search Efficiency

When testing 2 classifiers × 4 risk levels × 10 portfolios × 40 strategies = 3,200 combinations:

**Naive approach**: Create 3,200 independent containers
**Smart approach**: Organize hierarchically, compute shared components once

```
Classifier (computed once)
  └─ Risk Profiles
      └─ Portfolios  
          └─ Strategies (only these vary)
```

### 2. Flexible Search Questions

Different questions require different organizations:

```yaml
# Question: "How does my momentum strategy perform across conditions?"
hierarchy: [strategy, classifier, risk]  # Fix strategy, vary others

# Question: "What works best in bull markets?"
hierarchy: [classifier, risk, strategy]  # Fix classifier, vary strategies

# Question: "What's the optimal risk level?"
hierarchy: [risk, classifier, strategy]  # Fix risk, vary others
```

### 3. Clean Communication Without Coupling

Adapters separate routing from business logic:

```python
# Without adapters: Messy coupling
class Strategy:
    def process(self):
        signal = self.calculate()
        # Routing logic pollutes business logic
        if self.org == "strategy_first":
            for classifier in self.classifiers:
                classifier.receive(signal)
        elif self.org == "classifier_first":
            self.parent_classifier.receive(signal)

# With adapters: Clean separation
class Strategy:
    def process(self):
        signal = self.calculate()
        self.publish(signal)  # Adapter handles routing
```

## Practical Architecture Components

### 1. Containers: Business Logic Isolation
- Each container runs one piece of logic (strategy, classifier, risk manager)
- Containers can be organized hierarchically for efficiency
- Same container works in any hierarchy position

### 2. Adapters: Communication Patterns
- **Pipeline**: Sequential processing (the core flow)
- **Broadcast**: One-to-many distribution
- **Hierarchical**: Parent-child relationships
- **Selective**: Conditional routing

### 3. Events: Type-Safe Communication
- Strongly-typed events (no generic dictionaries)
- Full lineage tracking (correlation and causation)
- Schema versioning for evolution

## Real Example: Multi-Strategy Backtest

```yaml
# Define the combinatorial search
search:
  # Fixed: We're testing our momentum strategy
  strategy: 
    type: "momentum"
    params: {fast: 10, slow: 30}
  
  # Vary: Test across different conditions
  classifiers: ["hmm", "volatility", "pattern"]
  risk_levels: ["conservative", "balanced", "aggressive"]

# Adapters handle the communication
adapters:
  # Broadcast data to all test combinations
  - type: "broadcast"
    source: "market_data"
    targets: "${all_test_containers}"
  
  # Pipeline for execution flow
  - type: "pipeline"
    containers: ["strategy", "risk", "execution"]

# Result: 9 test runs with shared computation
# Not 9 separate backtests with duplicated work
```

## Key Benefits

### 1. Computational Efficiency
- Indicators computed once, shared by all strategies
- Classifiers computed once, shared by all children
- No duplicate work in combinatorial searches

### 2. Clean Architecture
- Business logic isolated in containers
- Communication patterns isolated in adapters
- Configuration drives behavior, not code

### 3. Flexible Research
- Easily test different combinations
- Invert hierarchies based on question
- Add new components without changing existing code

### 4. Production Ready
- Same code for backtest and live trading
- Type safety catches errors early
- Full observability through event tracking

## Summary

ADMF-PC is architecturally sophisticated to solve practical problems:

1. **It's a pipeline at heart** - The core flow is always BAR → SIGNAL → ORDER → FILL → UPDATE
2. **Hierarchies minimize computation** - Shared components computed once
3. **Adapters enable flexibility** - Change communication without changing logic
4. **Configuration drives everything** - No code changes for different searches

The complexity serves efficiency and flexibility, not abstraction for its own sake.
