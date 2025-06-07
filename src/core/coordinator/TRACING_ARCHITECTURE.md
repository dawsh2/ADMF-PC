# Container-Based Tracing Architecture

## Overview

This document describes how event tracing is implemented in ADMF-PC following the principle of clean separation between orchestration and event systems.

## Key Principles

1. **Orchestration knows nothing about events or tracing**
2. **Containers manage their own tracing based on configuration**
3. **Tracing is configured through execution settings, not orchestration**
4. **Metadata flows from orchestration for context only**

## Architecture

```
┌─────────────────────┐
│   User Config       │
│ (YAML/Dict)         │
│ ┌─────────────────┐ │
│ │ execution:      │ │
│ │   enable_tracing│ │
│ │   trace_settings│ │
│ └─────────────────┘ │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Orchestration      │ 
│  (Coordinator/      │
│   Sequencer)        │
│                     │
│ • NO event imports  │
│ • NO tracing config │
│ • Just metadata     │
└──────────┬──────────┘
           │ metadata only
           ▼
┌─────────────────────┐
│  Topology Builder   │
│                     │
│ • Reads user config │
│ • Passes to containers
│ • NO event knowledge│
└──────────┬──────────┘
           │ config + metadata
           ▼
┌─────────────────────┐
│    Containers       │
│                     │
│ • Check own config  │
│ • Create tracers    │
│ • Manage events     │
└─────────────────────┘
```

## Implementation Details

### 1. User Configuration

```yaml
# config/backtest_config.yaml
execution:
  enable_event_tracing: true
  trace_settings:
    max_events: 10000
    persist_to_disk: true
    trace_dir: ./traces
```

### 2. Orchestration Layer

```python
# Sequencer - NO event system knowledge
topology_definition = {
    'mode': 'backtest',
    'config': config,  # Contains tracing settings
    'metadata': {      # Just context, no tracing
        'workflow_id': 'wf_123',
        'phase_name': 'training'
    }
}
```

### 3. Container Implementation

```python
class Container:
    def __init__(self, config):
        # Container checks its own config
        if config.get('enable_tracing'):
            self._setup_tracing(config)
    
    def _setup_tracing(self, config):
        # Container creates its own tracer
        from ..events.tracing import EventTracer
        
        # Use metadata for trace naming
        metadata = config.get('metadata', {})
        trace_id = f"{metadata.get('workflow_id')}_{self.container_id}"
        
        self.event_tracer = EventTracer(trace_id)
        self.event_bus.subscribe_all(self.event_tracer.trace_event)
```

## Configuration Flow

1. **User** specifies tracing in execution settings
2. **Orchestration** passes config and metadata (no tracing knowledge)
3. **TopologyBuilder** includes tracing config when creating containers
4. **Containers** decide whether to trace based on their config
5. **EventTracer** created and managed entirely by containers

## Benefits

### Clean Separation
- Orchestration layer has zero event system dependencies
- Can test orchestration without any event infrastructure
- Clear architectural boundaries

### Flexibility
- Different containers can have different tracing configs
- Easy to enable/disable tracing per container type
- Tracing overhead only when needed

### Maintainability
- Tracing logic isolated in containers
- Easy to add new tracing features
- No coupling between layers

## Usage Examples

### Enable Tracing for All Containers
```yaml
execution:
  enable_event_tracing: true
  trace_settings:
    max_events: 10000
```

### Disable Tracing (Production)
```yaml
execution:
  enable_event_tracing: false
```

### Custom Tracing Per Container Type
```python
# In topology creation
portfolio_config = {
    'enable_tracing': config['execution']['enable_event_tracing'],
    'trace_settings': {
        **config['execution']['trace_settings'],
        'max_events': 50000  # More events for portfolios
    }
}
```

## Files Created

1. **container_with_tracing.py** - Example container implementation with tracing
2. **backtest_with_tracing.py** - Topology creation showing tracing flow
3. **sequencer_clean_v3.py** - Clean sequencer with no event knowledge
4. **example_backtest_with_tracing.yaml** - Configuration examples

## Migration from Old Architecture

### Before (Option 2)
```python
# Sequencer created EventTracer
phase_tracer = EventTracer(...)
topology_definition = {
    'event_tracer': phase_tracer  # Passed tracer object
}
```

### After (Option 3)
```python
# Sequencer knows nothing about tracing
topology_definition = {
    'metadata': {
        'workflow_id': workflow_id,
        'phase_name': phase_name
    }
}
# Containers create their own tracers
```

This architecture ensures complete separation of concerns while maintaining full tracing capabilities.