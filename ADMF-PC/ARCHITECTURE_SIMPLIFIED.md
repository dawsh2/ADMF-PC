# Simplified Architecture

## Problem: Deep Import Chain

The original architecture had a deep import chain:
```
main.py 
→ coordinator/__init__.py 
→ coordinator.py 
→ execution_modes.py 
→ execution/__init__.py 
→ backtest_engine.py 
→ strategy/__init__.py 
→ classifiers/__init__.py 
→ enhanced_classifier_container.py 
→ risk/protocols.py
```

This caused:
- Circular dependency risks
- Tight coupling between modules
- Slow startup times
- Difficult debugging

## Solution: Minimal Architecture

### Key Principles

1. **Lazy Loading**: Only import what's needed when it's needed
2. **Plugin Architecture**: Components register themselves
3. **Simple Interfaces**: Avoid complex hierarchies
4. **Minimal Dependencies**: Each module stands alone

### New Architecture

```
main.py
└── MinimalBootstrap (minimal_bootstrap.py)
    └── MinimalCoordinator (minimal_coordinator.py)
        └── SimpleBacktestEngine (lazy loaded only when needed)
```

### Benefits

1. **No Deep Imports**: Maximum 3 levels deep
2. **Fast Startup**: Only load what's needed
3. **Easy Testing**: Each component is independent
4. **Clear Flow**: Easy to understand execution path
5. **Extensible**: Easy to add new workflow types

### Implementation Details

#### MinimalBootstrap
- No imports in __init__
- Lazy coordinator creation
- Simple interface

#### MinimalCoordinator  
- Self-contained workflow types
- Lazy engine imports
- Plugin-based manager registration

#### Execution Flow

```python
# 1. Main creates bootstrap (no imports)
bootstrap = MinimalBootstrap()

# 2. Bootstrap creates coordinator (lazy)
coordinator = bootstrap.create_coordinator()

# 3. Coordinator executes workflow
result = await coordinator.execute_workflow(config)

# 4. Only now is the engine imported
from ...execution.simple_backtest_engine import SimpleBacktestEngine
```

### Configuration

The same YAML configuration works with both architectures:
```yaml
workflow_type: backtest
data:
  file_path: data/SYNTH_1min.csv
backtest:
  strategies:
    - type: price_threshold
      parameters:
        buy_threshold: 90.0
        sell_threshold: 100.0
```

### Migration Path

1. **Phase 1**: Use MinimalBootstrap as default (current)
2. **Phase 2**: Refactor existing modules to follow minimal principles
3. **Phase 3**: Deprecate deep import chains
4. **Phase 4**: Full plugin architecture

### Best Practices

1. **Avoid Deep Imports**: Keep import chains under 3 levels
2. **Use Lazy Loading**: Import only when executing, not at module level
3. **Define Interfaces**: Use protocols/ABCs for clean contracts
4. **Minimize Dependencies**: Each module should work independently
5. **Plugin Architecture**: Components register themselves with coordinators