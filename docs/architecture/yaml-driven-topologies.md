# YAML-Driven Topology System

## Overview

The YAML-driven topology system allows users to define complete trading system configurations using only YAML files, without writing any Python code for topology creation. This makes the system more accessible and easier to configure.

## Architecture

### 1. Pattern Files

Topology patterns are stored as YAML files in `src/core/coordinator/patterns/`:

```yaml
# Pattern structure
name: pattern_name
description: What this pattern does

components:
  - type: component_type
    from_config: config_path

containers:
  - name: container_name
    type: container_type
    config:
      key: value

routes:
  - name: route_name
    type: route_type
    source: source_container
    target: target_container

behaviors:
  - type: behavior_type
    config: behavior_config
```

### 2. Generic TopologyBuilder

The `TopologyBuilder` class interprets patterns and builds topologies:

```python
builder = TopologyBuilder()
topology = builder.build_topology({
    'mode': 'backtest',
    'config': user_config,
    'tracing_config': {...}
})
```

### 3. Value Resolution

The builder supports several value resolution patterns:

#### Template Strings
```yaml
name_template: "{symbol}_{timeframe}_data"
```

#### Config References
```yaml
config:
  symbol:
    from_config: symbols
    default: SPY
```

#### Context References
```yaml
target: "$root_event_bus"
```

#### Foreach Loops
```yaml
foreach:
  symbol:
    from_config: symbols
  timeframe:
    from_config: timeframes
```

## Pattern Examples

### Backtest Pattern

Creates the full pipeline: data → features → strategies → portfolios → risk → execution

```yaml
name: backtest
description: Full backtest pipeline

containers:
  # Data containers for each symbol/timeframe
  - name_template: "{symbol}_{timeframe}_data"
    type: data
    foreach:
      symbol: {from_config: symbols}
      timeframe: {from_config: timeframes}
    
  # Portfolio containers for each strategy/risk combo
  - name_template: "portfolio_{combo_id}"
    type: portfolio
    foreach:
      combo: "$generated.parameter_combinations"
```

### Signal Generation Pattern

Generates and saves trading signals:

```yaml
name: signal_generation
description: Generate and save signals

routes:
  - name: signal_saver
    type: signal_saver
    save_directory: {from_config: signal_save_directory}
```

### Signal Replay Pattern

Replays saved signals through execution:

```yaml
name: signal_replay
description: Replay saved signals

containers:
  - name: signal_replay
    type: signal_replay
    config:
      signal_directory: {from_config: signal_directory}
```

## User Workflow

### 1. Basic Usage

Users only need to provide configuration:

```python
config = {
    'symbols': ['SPY', 'QQQ'],
    'strategies': [
        {'type': 'momentum', 'fast_period': 10}
    ],
    'risk_profiles': [
        {'type': 'conservative', 'max_position_size': 0.1}
    ]
}

builder = TopologyBuilder()
topology = builder.build_topology({
    'mode': 'backtest',
    'config': config
})
```

### 2. Custom Patterns

Users can create their own patterns:

```yaml
# my_patterns/trend_following.yaml
name: trend_following
description: Custom trend following system

containers:
  # Custom container configuration
  
behaviors:
  # Custom behaviors
```

Then use them:

```python
builder.patterns['trend_following'] = load_yaml('my_patterns/trend_following.yaml')
topology = builder.build_topology({'mode': 'trend_following', 'config': {...}})
```

### 3. Pure YAML Workflow

For maximum simplicity, users can define everything in YAML:

```yaml
# my_backtest.yaml
mode: backtest
config:
  symbols: [SPY, QQQ]
  timeframes: [5T, 15T]
  strategies:
    - type: momentum
      fast_period: 10
      slow_period: 30
  risk_profiles:
    - type: conservative
      max_position_size: 0.1
```

Then run:

```bash
python -m admfpc run my_backtest.yaml
```

## Benefits

1. **No Code Required**: Users can configure complete systems without Python
2. **Declarative**: Patterns describe what to build, not how
3. **Reusable**: Patterns can be shared and versioned
4. **Extensible**: New patterns can be added without modifying core code
5. **Testable**: Patterns can be validated before execution

## Advanced Features

### Parameter Combinations

The builder automatically generates parameter combinations:

```yaml
foreach:
  combo: "$generated.parameter_combinations"
```

This creates all combinations of strategies × risk profiles.

### Pattern Matching

Use wildcards to match multiple containers:

```yaml
routes:
  - type: broadcast
    source: execution
    targets: "portfolio_*"  # Matches all portfolio containers
```

### Conditional Creation

Use config presence to conditionally create components:

```yaml
containers:
  - name: risk_manager
    type: risk
    if: {from_config: enable_risk_management, default: true}
```

### Behaviors

Special behaviors handle complex routing patterns:

```yaml
behaviors:
  - type: feature_dispatcher
    source_pattern: "*_features"
    target: strategies
```

## Migration Guide

To migrate existing Python topologies to YAML:

1. Identify the pattern (containers, routes, behaviors)
2. Convert imperative code to declarative specifications
3. Test with the same configuration
4. Remove the old Python topology file

Example migration:

```python
# OLD: Python code
for symbol in symbols:
    container = factory.create_container(...)
    containers[f"{symbol}_data"] = container
```

```yaml
# NEW: YAML pattern
containers:
  - name_template: "{symbol}_data"
    foreach:
      symbol: {from_config: symbols}
```

## Future Enhancements

1. **Pattern Inheritance**: Patterns extending other patterns
2. **Pattern Composition**: Combining multiple patterns
3. **Visual Editor**: GUI for creating patterns
4. **Pattern Validation**: Schema validation for patterns
5. **Pattern Library**: Repository of community patterns