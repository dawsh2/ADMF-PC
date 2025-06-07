# Routing Directory Cleanup Plan

## Step 1: Rename helpers directory
```bash
mv src/core/coordinator/topologies/helpers src/core/coordinator/topologies/route_builders
```

## Step 2: Create new routing directory structure
```bash
# Create organized subdirectories
mkdir -p src/core/routing/core
mkdir -p src/core/routing/specialized  
mkdir -p src/core/routing/experimental

# Move core routes
mv src/core/routing/pipeline_route_protocol.py src/core/routing/core/pipeline.py
mv src/core/routing/broadcast_route.py src/core/routing/core/broadcast.py
mv src/core/routing/selective_route.py src/core/routing/core/selective.py

# Move specialized routes
mv src/core/routing/risk_service_route.py src/core/routing/specialized/
mv src/core/routing/execution_service_route.py src/core/routing/specialized/

# Move experimental routes
mv src/core/routing/hierarchical_route.py src/core/routing/experimental/
```

## Step 3: Create missing SignalSaverRoute

Since signal_saver is referenced but not implemented, we need to create it:

```python
# src/core/routing/specialized/signal_saver.py
class SignalSaverRoute:
    """Route that captures and saves signals to disk."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.save_directory = config.get('save_directory', './signals')
        self.root_event_bus = config.get('root_event_bus')
        # Implementation to save signals...
```

## Step 4: Update imports in factory.py

Update to use new paths:
```python
# Core routes
from .core.pipeline import PipelineRoute
from .core.broadcast import BroadcastRoute  
from .core.selective import SelectiveRoute

# Specialized routes
from .specialized.risk_service import RiskServiceRoute
from .specialized.signal_saver import SignalSaverRoute
```

## Step 5: Remove unused code

These files contain complex variants we're not using:
- Create variants.py in experimental/ to consolidate:
  - create_conditional_pipeline
  - create_parallel_pipeline
  - create_filtered_broadcast
  - create_priority_broadcast
  - create_aggregating_hierarchy
  - create_filtered_hierarchy
  - create_capability_based_router
  - create_load_balanced_router
  - create_content_based_router
  - FanOutRoute

## Step 6: Improve signal flow routing

Instead of Feature Dispatcher component, use proper routes:

1. **Feature → Strategy**: Use broadcast with filtering
2. **Strategy → Portfolio**: Use selective routing based on combo_id

## Benefits

1. **Clearer organization**: Core vs specialized vs experimental
2. **Reduced complexity**: Only keep what we use
3. **Better naming**: No more "helpers" code smell
4. **Proper routing**: Replace custom components with standard routes
5. **Extensibility**: Experimental folder for future patterns