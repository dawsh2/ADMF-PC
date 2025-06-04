# Workflow Module Consolidation Cleanup Plan

## Current State Analysis

After implementing the pattern-based architecture, we now have **duplicate functionality** that should be consolidated:

### Files Currently in workflows/
```
src/core/coordinator/workflows/
├── __init__.py                    # ✅ KEEP - exports standard interfaces
├── workflow_manager.py            # ✅ KEEP - canonical orchestrator
├── backtest.py                    # 🔄 LEGACY - evaluate for deprecation
├── container_factories.py        # ❌ REMOVE - 871 lines of duplication
├── containers_pipeline.py        # ❌ REMOVE - thin wrapper around container_factories
├── modes/                         # 🔄 EVALUATE - may have useful config patterns
│   ├── backtest.py               # 🔄 EVALUATE - merge useful parts
│   └── signal_replay.py          # 🔄 EVALUATE - merge useful parts
└── feature_hub_workflow.py       # 🔄 EVALUATE - feature computation patterns
```

### Core Infrastructure (Keep These)
```
src/core/containers/factory.py    # ✅ CANONICAL - container creation
src/core/communication/factory.py # ✅ CANONICAL - adapter creation  
```

## Consolidation Plan

### Phase 1: Remove Duplicate Factories ✅ HIGH PRIORITY

#### Remove These Files
```bash
# These are pure duplication of core factory functionality
rm src/core/coordinator/workflows/container_factories.py     # 871 lines
rm src/core/coordinator/workflows/containers_pipeline.py    # 289 lines  
```

**Rationale**: 
- `container_factories.py` duplicates `core/containers/factory.py` functionality
- `containers_pipeline.py` is just a thin wrapper around `container_factories.py`
- WorkflowManager now uses core factories directly via delegation

#### Impact Analysis
```python
# Files that import from container_factories.py:
find src/ -name "*.py" -exec grep -l "from.*container_factories import\|import.*container_factories" {} \;

# Need to update these imports to use:
from ...containers.factory import get_global_factory, register_container_type
# OR
from .workflow_manager import WorkflowManager  # for workflow orchestration
```

### Phase 2: Create Unified Workflow Factory 🔄 MEDIUM PRIORITY

Instead of multiple factory files, create **one workflow factory** that coordinates both:

```python
# src/core/coordinator/workflows/factory.py
"""
Unified Workflow Factory

This is the SINGLE factory for creating complete workflows.
It delegates to core factories but provides workflow-level orchestration.
"""

from typing import Dict, Any, Optional
from ...containers.factory import get_global_factory
from ...communication.factory import AdapterFactory
from .workflow_manager import WorkflowManager

class WorkflowFactory:
    """
    Unified factory for creating complete workflows.
    
    This factory:
    1. Uses core/containers/factory for container creation
    2. Uses core/communication/factory for adapter creation  
    3. Provides workflow-level orchestration and patterns
    """
    
    def __init__(self):
        self.container_factory = get_global_factory()
        self.communication_factory = AdapterFactory()
        self.workflow_manager = WorkflowManager()
    
    def create_workflow(self, pattern_name: str, config: Dict[str, Any]) -> WorkflowManager:
        """Create complete workflow with containers + communication."""
        # Delegate to WorkflowManager which coordinates both factories
        return self.workflow_manager
    
    def create_containers_only(self, pattern_name: str, config: Dict[str, Any]):
        """Create just containers (delegates to core factory)."""
        return self.container_factory.compose_pattern(pattern_name, config)
    
    def create_communication_only(self, comm_config: List[Dict[str, Any]], containers: Dict):
        """Create just communication (delegates to core factory)."""
        return self.communication_factory.create_adapters_from_config(comm_config, containers)

# Convenience functions
def create_workflow(pattern_name: str, config: Dict[str, Any]) -> WorkflowManager:
    """Convenience function for workflow creation."""
    factory = WorkflowFactory()
    return factory.create_workflow(pattern_name, config)
```

### Phase 3: Evaluate Legacy Files 🔄 LOW PRIORITY

#### backtest.py (Legacy Workflow)
**Action**: Evaluate if BacktestWorkflow adds value over WorkflowManager
```python
# Current: src/core/coordinator/workflows/backtest.py
class BacktestWorkflow:  # 290 lines
    # Does this provide functionality not in WorkflowManager?
    
# Option 1: Deprecate if WorkflowManager covers all cases
# Option 2: Refactor as thin wrapper around WorkflowManager
class BacktestWorkflow:
    def __init__(self, config):
        self.workflow_manager = WorkflowManager()
    
    def execute(self):
        return await self.workflow_manager.execute(self._convert_config())
```

#### modes/ Directory
**Action**: Extract useful configuration patterns, then remove
```python
# modes/backtest.py and modes/signal_replay.py may have useful:
# - Configuration schemas
# - Default parameter sets  
# - Validation logic

# Extract these to WorkflowManager pattern definitions:
self._workflow_patterns = {
    'simple_backtest': {
        'default_config': extract_from_modes_backtest(),  # Useful defaults
        'validation': extract_validation_logic(),          # Useful validation
        # ...
    }
}
```

#### feature_hub_workflow.py
**Action**: Integrate feature computation patterns into WorkflowManager
```python
# This file has useful feature computation patterns
# Integrate into WorkflowManager communication configs:

def _get_feature_hub_communication(self, containers):
    """Communication pattern for centralized feature computation."""
    return [{
        'name': 'feature_hub_pattern',
        'type': 'broadcast',
        'source': 'FeatureHubContainer',
        'targets': ['StrategyContainer_1', 'StrategyContainer_2'],
        'event_types': ['FEATURE']
    }]
```

## Consolidation Commands

### Step 1: Remove Duplicate Factories
```bash
# Backup first
cp src/core/coordinator/workflows/container_factories.py /tmp/container_factories_backup.py
cp src/core/coordinator/workflows/containers_pipeline.py /tmp/containers_pipeline_backup.py

# Remove duplicates
rm src/core/coordinator/workflows/container_factories.py
rm src/core/coordinator/workflows/containers_pipeline.py

# Update imports in affected files
find src/ -name "*.py" -exec sed -i 's/from \.container_factories import/from ...containers.factory import get_global_factory; factory = get_global_factory(); # Updated import/g' {} \;
find src/ -name "*.py" -exec sed -i 's/from \.containers_pipeline import/from .workflow_manager import WorkflowManager/g' {} \;
```

### Step 2: Update __init__.py
```python
# src/core/coordinator/workflows/__init__.py
"""
Coordinator workflows for ADMF-PC.

ARCHITECTURE: Uses pattern-based architecture with clear factory separation.
- Container creation → core.containers.factory
- Communication setup → core.communication.factory  
- Workflow orchestration → workflow_manager.py
"""

# Primary interface (RECOMMENDED)
from .workflow_manager import WorkflowManager, ComposableWorkflowManagerPipeline

# Core factory functions (for direct use)
from ...containers.factory import (
    get_global_factory,
    get_global_registry, 
    compose_pattern,
    register_container_type
)

# Legacy workflows (DEPRECATED - use WorkflowManager instead)
from .backtest import BacktestWorkflow

__all__ = [
    # Primary interface
    'WorkflowManager',
    'ComposableWorkflowManagerPipeline',
    
    # Core factory access
    'get_global_factory',
    'get_global_registry',
    'compose_pattern', 
    'register_container_type',
    
    # Legacy (deprecated)
    'BacktestWorkflow',
]
```

### Step 3: Create Migration Guide
```python
# MIGRATION_GUIDE.md

# OLD (deprecated):
from .container_factories import create_data_container, create_strategy_container
data_container = create_data_container(config)

# NEW (standard):
from ...containers.factory import get_global_factory
factory = get_global_factory()
containers = factory.compose_pattern('simple_backtest', config)

# OR (workflow-level):
from .workflow_manager import WorkflowManager
manager = WorkflowManager()
result = await manager.execute(workflow_config)
```

## Expected Results

### Before Cleanup
```
workflows/
├── __init__.py (64 lines)
├── workflow_manager.py (400+ lines) ✅ CANONICAL
├── backtest.py (290 lines) 🔄 LEGACY
├── container_factories.py (871 lines) ❌ DUPLICATE  
├── containers_pipeline.py (289 lines) ❌ DUPLICATE
├── modes/ (2 files) 🔄 EVALUATE
└── feature_hub_workflow.py (585 lines) 🔄 EVALUATE

Total: ~2500 lines with significant duplication
```

### After Cleanup
```
workflows/
├── __init__.py (30 lines) ✅ CLEAN EXPORTS
├── workflow_manager.py (400+ lines) ✅ CANONICAL
└── factory.py (50 lines) ✅ UNIFIED INTERFACE (optional)

Total: ~480 lines, no duplication
Reduction: 80% fewer lines, 100% less duplication
```

### Benefits
✅ **Single Source of Truth**: Each pattern defined once in WorkflowManager  
✅ **No Duplication**: Uses core factories via delegation  
✅ **Clear Architecture**: Obvious separation of responsibilities  
✅ **Easier Maintenance**: Changes happen in one place  
✅ **Better Performance**: Fewer redundant imports and objects  

## Implementation Priority

1. **HIGH**: Remove container_factories.py and containers_pipeline.py (immediate 1160 line reduction)
2. **MEDIUM**: Create unified workflow factory.py (clean interface)
3. **LOW**: Evaluate and migrate/remove legacy files (modes/, backtest.py, feature_hub_workflow.py)

This cleanup will complete our transition to the pattern-based architecture standard while eliminating all duplication.