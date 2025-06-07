# Trace Level Implementation Summary

## Overview

We have successfully implemented trace level presets for the ADMF-PC event tracing system. This allows users to easily control event tracing overhead with simple preset names instead of manually configuring trace settings.

## What Was Implemented

### 1. Trace Level Presets Module
Created `src/core/events/tracing/trace_levels.py` with:
- `TraceLevel` enum: NONE, MINIMAL, NORMAL, DEBUG
- Preset configurations for each level
- Functions to apply trace levels to configuration
- Container-specific overrides per trace level

### 2. Orchestration Chain Updates

#### Coordinator (`src/core/coordinator_refactor/coordinator.py`)
- Added `_apply_trace_level_config()` method
- Automatically applies trace level presets before workflow execution
- Transforms simple `trace_level: minimal` into full execution config

#### Sequencer (`src/core/coordinator_refactor/sequencer.py`)
- Updated to extract trace configuration from execution config
- Passes trace settings to TopologyBuilder via `tracing_config`
- Maintains trace_id with workflow and phase information

#### TopologyBuilder (`src/core/coordinator_refactor/topology.py`)
- Converts `tracing_config` into container-readable `execution` config
- Ensures trace settings flow to all containers
- Preserves container-specific settings from trace levels

### 3. Container Integration
The existing container implementation (`src/core/containers/container.py`) already:
- Has smart defaults (portfolio containers always track metrics)
- Reads trace configuration from execution config
- Conditionally enables tracing based on container role and settings

## Trace Level Presets

### NONE (Production)
```yaml
trace_level: none
```
- Tracing completely disabled
- Zero overhead
- No event storage

### MINIMAL (Optimization)
```yaml
trace_level: minimal
```
- Portfolio containers only - track open trades
- No max event limit - retention policy manages memory
- Events deleted when trades close
- Metrics persist: win rate, total return, P&L
- **Tradeoff**: Max drawdown = 0 (no equity curve)
- All other containers: tracing disabled

### NORMAL (Development)
```yaml
trace_level: normal
```
- Max 10,000 events retained
- Trade-complete retention policy
- Data containers limited to 1,000 events
- Good balance of visibility and performance

### DEBUG (Full Tracing)
```yaml
trace_level: debug
```
- Max 50,000 events retained
- All events traced
- Full timing information
- Maximum visibility for debugging

## Usage Examples

### Simple Usage
```yaml
workflow: simple_backtest
trace_level: minimal  # That's it!

data:
  symbols: [SPY]
  # ... rest of config
```

### Custom Override
```yaml
workflow: complex_workflow
trace_level: normal  # Base level

# Override for specific needs
execution:
  trace_settings:
    container_settings:
      portfolio_001:
        max_events: 20000  # More for this container
```

### Multi-Phase Workflows
Each phase can have different trace levels:
```yaml
workflow: adaptive_ensemble
trace_level: minimal  # Default for all phases

phases:
  - name: optimization
    # Inherits minimal
  - name: validation
    config_override:
      trace_level: normal  # More tracing for validation
```

## Benefits

1. **Simplicity**: Users just specify `trace_level: minimal` instead of complex configuration
2. **Smart Defaults**: Each level is optimized for its use case
3. **Memory Efficiency**: Optimization workflows use minimal memory
4. **Flexibility**: Can still override specific settings when needed
5. **Container Awareness**: Different container types get appropriate limits

## Implementation Details

### Configuration Flow
```
User Config (trace_level: minimal)
    ↓
Coordinator._apply_trace_level_config()
    ↓
Sequencer (passes to topology via tracing_config)
    ↓
TopologyBuilder (converts to execution config)
    ↓
Containers (read execution.trace_settings)
```

### Backward Compatibility
- Existing configurations without trace_level continue to work
- Manual execution.trace_settings configurations are preserved
- Container smart defaults remain active

## Testing

Created `examples/trace_level_example.py` demonstrating:
- Optimization with minimal tracing
- Debugging with full tracing
- Production with no tracing
- Custom configurations
- Multi-phase workflows

## Next Steps

The trace level system is fully implemented and integrated. Users can now:

1. Use simple trace level presets for common scenarios
2. Override specific settings when needed
3. Let the system handle the complexity of trace configuration

The implementation follows ADMF-PC principles:
- Single source of truth (trace_levels.py)
- Configuration-driven behavior
- Smart defaults with override capability
- Clean separation of concerns