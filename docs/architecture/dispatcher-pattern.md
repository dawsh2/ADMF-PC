# The Dispatcher Pattern

## Overview

The Coordinator functions as a workflow dispatcher that sequences operations according to configuration specifications. This design abstracts sequencing logic into a dedicated module, allowing other components to focus on their domain-specific responsibilities.

## Core Philosophy

The dispatcher embodies a crucial design principle: sophisticated behavior should emerge from simple orchestration of well-defined components, rather than from intelligent coordination logic. The Coordinator operates as a straightforward execution manager that follows specified sequences without interpretation. This design choice prevents the Coordinator from becoming an over-complex "god module" while distributing system responsibilities appropriately across modular components.

This approach eliminates a common source of non-determinism in trading systems—coordinators that make "helpful" optimizations or adaptations that make results difficult to reproduce. In ADMF-PC, identical configurations produce identical execution sequences, regardless of system load, available resources, or previous executions. The modular architecture enables this simplicity by ensuring each component handles its own domain logic while the Coordinator focuses solely on sequencing operations as specified.

## Simple Backtest Dispatch

```
Configuration: simple_backtest.yaml
┌─────────────────────────────────────────────────────────────┐
│  data:                                                      │
│    source: csv                                              │
│    symbols: [SPY]                                           │
│  strategy:                                                  │
│    type: momentum                                           │
│    parameters: {lookback: 20}                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dispatcher Sequence                      │
│  1. Initialize DataContainer(csv, SPY)                      │
│  2. Initialize StrategyContainer(momentum, lookback=20)     │
│  3. Create event bindings: Data → Strategy                  │
│  4. Execute workflow until completion                       │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Phase Optimization Dispatch

```
Configuration: parameter_optimization.yaml
┌─────────────────────────────────────────────────────────────┐
│  phases:                                                    │
│    - parameter_discovery:                                   │
│        mode: signal_generation  # Generate & save signals   │
│        grid: {lookback: [10,20,30], threshold: [0.01,0.02]} │
│    - regime_analysis:                                       │
│        mode: analysis          # Read results, no execution │
│        classifiers: [volatility, momentum]                  │
│    - ensemble_optimization:                                 │
│        mode: signal_replay     # Replay saved signals       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Multi-Phase Dispatch Sequence                │
│                                                             │
│  Phase 1: Parameter Discovery (Signal Generation Mode)      │
│  ├─ For each parameter combination:                         │
│  │  ├─ Initialize signal generation containers              │
│  │  ├─ Generate signals WITHOUT execution                   │
│  │  ├─ Save signals to disk (signals/trial_N.jsonl)        │
│  │  └─ Record basic performance estimates                   │
│  └─ Output: Saved signals for all parameter combinations    │
│                                                             │
│  Phase 2: Regime Analysis (Analysis Mode)                   │
│  ├─ Read Phase 1 performance estimates                      │
│  ├─ Classify market regimes from historical data           │
│  └─ Identify best parameters per regime                     │
│                                                             │
│  Phase 3: Ensemble Optimization (Signal Replay Mode)        │
│  ├─ Load signal logs from Phase 1 (10-100x faster!)        │
│  ├─ Test thousands of weight combinations                   │
│  ├─ No indicator recalculation needed                      │
│  └─ Generate optimal ensemble weights                       │
└─────────────────────────────────────────────────────────────┘
```

### Why This Pattern Matters

The signal generation → signal replay pattern demonstrates ADMF-PC's key performance optimization:

1. **Phase 1 (Signal Generation)**: Expensive computation happens once
   - Calculate all indicators for each parameter set
   - Generate trading signals
   - Save signals to disk
   - ~1 minute per parameter combination

2. **Phase 3 (Signal Replay)**: Rapid optimization on saved signals
   - Load pre-generated signals
   - Test weight combinations without recalculation
   - 10-100x faster than regenerating signals
   - Test 10,000 weight combinations in the time it takes to generate 100 signal sets

### Practical Example: Performance Impact

```
Traditional Approach (Full Backtest for Everything):
- 6 parameter combinations × 1 minute each = 6 minutes
- 1000 weight combinations × 1 minute each = 1000 minutes (16.7 hours)
- Total: ~17 hours

ADMF-PC Approach (Signal Generation + Replay):
- Phase 1: 6 parameter combinations × 1 minute = 6 minutes (signal generation)
- Phase 3: 1000 weight combinations × 0.01 minutes = 10 minutes (signal replay)
- Total: ~16 minutes (63x faster!)

For larger parameter spaces:
- 100 parameters, 10,000 weight combinations
- Traditional: 100 + 10,000 = 10,100 minutes (168 hours / 7 days)
- ADMF-PC: 100 + 100 = 200 minutes (3.3 hours) - 50x faster!
```

This architectural pattern enables researchers to explore vastly larger parameter and weight spaces in practical timeframes.

## Key Benefits of the Dispatcher Pattern

### 1. Deterministic Execution

The dispatcher's lack of state awareness becomes advantageous—it executes each phase exactly as specified without attempting to optimize based on intermediate results, ensuring consistent execution paths across different runs.

### 2. Separation of Concerns

- **Coordinator**: Manages workflow sequencing
- **Containers**: Handle domain-specific logic
- **Components**: Implement business logic

This clean separation makes the system easier to understand, test, and maintain.

### 3. Workflow Transparency

Configuration files become complete specifications of execution flow. Anyone reading a configuration can understand exactly what will happen without examining code.

### 4. Error Isolation

When errors occur, the dispatcher pattern makes it clear whether the issue is in:
- Workflow sequencing (dispatcher)
- Container initialization (container factory)
- Business logic (components)

## Implementation Patterns

### Sequential Execution

```python
class Dispatcher:
    def execute_sequential(self, phases: List[Phase]) -> None:
        for phase in phases:
            container = self.factory.create_container(phase.type)
            result = container.execute(phase.config)
            self.workspace.save_result(phase.name, result)
```

### Parallel Execution

```python
class Dispatcher:
    def execute_parallel(self, phases: List[Phase]) -> None:
        with ProcessPoolExecutor() as executor:
            futures = []
            for phase in phases:
                future = executor.submit(self.execute_phase, phase)
                futures.append((phase.name, future))
            
            for name, future in futures:
                result = future.result()
                self.workspace.save_result(name, result)
```

### Conditional Execution

```python
class Dispatcher:
    def execute_conditional(self, phases: List[Phase]) -> None:
        for phase in phases:
            if self.evaluate_condition(phase.condition):
                container = self.factory.create_container(phase.type)
                result = container.execute(phase.config)
                self.workspace.save_result(phase.name, result)
```

## Avoiding Common Pitfalls

### ❌ Don't: Smart Coordination

```python
# Bad: Coordinator making decisions
class SmartCoordinator:
    def execute(self, phases):
        for phase in phases:
            # Coordinator shouldn't make optimization decisions
            if self.should_skip_phase(phase):  
                continue
            
            # Coordinator shouldn't modify configurations
            optimized_config = self.optimize_config(phase.config)
            self.execute_phase(phase, optimized_config)
```

### ✅ Do: Simple Dispatch

```python
# Good: Coordinator just dispatches
class SimpleDispatcher:
    def execute(self, phases):
        for phase in phases:
            # Execute exactly as specified
            container = self.factory.create(phase.type)
            container.execute(phase.config)
```

### ❌ Don't: State in Dispatcher

```python
# Bad: Dispatcher maintaining state
class StatefulDispatcher:
    def __init__(self):
        self.execution_history = []
        self.performance_cache = {}
        
    def execute(self, phase):
        # Dispatcher shouldn't track history
        if phase in self.execution_history:
            return self.performance_cache[phase]
```

### ✅ Do: Stateless Dispatch

```python
# Good: Dispatcher is stateless
class StatelessDispatcher:
    def execute(self, phase):
        # Each execution is independent
        container = self.factory.create(phase.type)
        return container.execute(phase.config)
```

## Advanced Patterns

### Workflow Composition

The dispatcher pattern enables workflows to be composed from simpler workflows:

```yaml
composite_workflow:
  phases:
    - name: "morning_analysis"
      workflow: "configs/morning_workflow.yaml"
      
    - name: "strategy_selection"
      workflow: "configs/strategy_selector.yaml"
      depends_on: ["morning_analysis"]
      
    - name: "risk_adjustment"
      workflow: "configs/risk_workflow.yaml"
      depends_on: ["strategy_selection"]
```

### Event-Driven Dispatch

The dispatcher can respond to external events while maintaining simplicity:

```python
class EventDrivenDispatcher:
    def __init__(self):
        self.event_bus = EventBus()
        self.event_bus.subscribe("MARKET_OPEN", self.on_market_open)
        self.event_bus.subscribe("DATA_AVAILABLE", self.on_data_available)
        
    def on_market_open(self, event):
        # Still just dispatching, not making decisions
        self.execute_workflow("configs/market_open_workflow.yaml")
        
    def on_data_available(self, event):
        # Configuration determines what to do with new data
        config = self.load_config("configs/data_processing.yaml")
        self.execute_workflow(config)
```

## Testing the Dispatcher

The simplicity of the dispatcher pattern makes testing straightforward:

```python
def test_dispatcher_sequence():
    """Test that dispatcher executes phases in order"""
    dispatcher = Dispatcher()
    executed_phases = []
    
    # Mock factory that tracks execution
    dispatcher.factory = MockFactory(executed_phases)
    
    config = {
        "phases": [
            {"name": "phase1", "type": "backtest"},
            {"name": "phase2", "type": "analysis"},
            {"name": "phase3", "type": "report"}
        ]
    }
    
    dispatcher.execute(config)
    
    assert executed_phases == ["phase1", "phase2", "phase3"]
```

## Summary

The dispatcher pattern is fundamental to ADMF-PC's architecture because it:

1. **Ensures Reproducibility**: Same configuration always produces same execution
2. **Simplifies Debugging**: Clear separation between coordination and logic
3. **Enables Composition**: Complex workflows from simple building blocks
4. **Maintains Flexibility**: New workflow patterns without code changes
5. **Facilitates Testing**: Simple dispatch logic is easy to verify

The key insight is that coordination complexity should come from configuration, not code. The dispatcher remains simple while enabling sophisticated workflow patterns through composition.