# Core Module File Review Checklist

## Review Criteria
For each file, check:
- [ ] **Purpose**: Is the file's purpose clear and necessary?
- [ ] **ADMF-PC Compliance**: No inheritance (except from Protocol/ABC), uses composition
- [ ] **Imports**: No imports from removed modules (dependencies/, infrastructure/, logging/)
- [ ] **Naming**: No "enhanced_", "improved_", etc. in filenames
- [ ] **Simplicity**: Could this be simpler or combined with another file?
- [ ] **Usage**: Is this file actually used by the system?
- [ ] **Future Value**: If not currently used, is it providing future value or just speculative complexity?

---

## 1. core/types/ Directory

### Files to Review:
- [ ] `__init__.py` - Type exports and organization
- [ ] `decimal.py` - Financial calculation utilities (moved from utils/)
- [ ] `duck_types.py` - Flexible duck-typed interfaces
- [ ] `events.py` - Event system types
- [ ] `trading.py` - Trading domain types (Order, Signal, Position)
- [ ] `workflow.py` - Workflow and coordination types

### Key Questions:
- Are all type definitions actually used?
- Should decimal utilities stay here or move elsewhere?
- Are duck types providing value or adding complexity?

---

## 2. core/events/ Directory

### Files to Review:
- [ ] `__init__.py` - Event system exports
- [ ] `event_bus.py` - Core event bus implementation
- [ ] `isolation.py` - Container event isolation
- [ ] `semantic.py` - Semantic event types
- [ ] `subscription_manager.py` - Event subscription management
- [ ] `type_flow_analysis.py` - Event type flow analysis
- [ ] `type_flow_integration.py` - Type flow integration
- [ ] `type_flow_visualization.py` - Type flow visualization

### Subdirectory: events/tracing/
- [ ] `__init__.py`
- [ ] `adapter_integration.py`
- [ ] `coordinator_integration.py`
- [ ] `event_store.py`
- [ ] `event_tracer.py`
- [ ] `storage_backends.py`
- [ ] `traced_event.py`
- [ ] `traced_event_bus.py`

### Key Questions:
- Is the tracing subdirectory necessary or over-engineered?
- Are semantic events actually used?
- Is type flow analysis providing value?

---

## 3. core/components/ Directory

### Files to Review:
- [ ] `__init__.py` - Component framework exports
- [ ] `discovery.py` - Component discovery mechanism
- [ ] `factory.py` - Component factory
- [ ] `protocols.py` - Component protocols
- [ ] `registry.py` - Component registry

### Key Questions:
- Is component discovery actually used or needed?
- Could factory and registry be combined?
- Are all protocols being implemented?

---

## 4. core/containers/ Directory

### Files to Review:
- [ ] `__init__.py` - Container exports
- [ ] `container.py` - Canonical container implementation
- [ ] `factory.py` - Container factory (renamed from composition_engine)
- [ ] `naming.py` - Container naming utilities
- [ ] `protocols.py` - Container protocols

### Subdirectory: containers/backtest/
- [ ] `__init__.py` - Backtest-specific container code?

### Key Questions:
- Is naming.py necessary or could it be part of container.py?
- What's in the backtest subdirectory?
- Are all container protocols implemented?

---

## 5. core/communication/ Directory

### Files to Review:
- [ ] `__init__.py` - Communication exports
- [ ] `broadcast_adapter.py` - Broadcast communication pattern
- [ ] `factory.py` - Adapter factory
- [ ] `helpers.py` - Communication helper functions
- [ ] `hierarchical_adapter.py` - Hierarchical communication pattern
- [ ] `integration.py` - Legacy integration layer
- [ ] `pipeline_adapter_protocol.py` - Pipeline communication pattern
- [ ] `protocols.py` - Communication protocols
- [ ] `selective_adapter.py` - Selective routing pattern
- [ ] `IMPLEMENTATION_SUMMARY.md` - Documentation
- [ ] `MIGRATION_GUIDE.md` - Documentation
- [ ] `README.md` - Documentation

### Key Questions:
- Are all adapter patterns actually used?
- Is integration.py still needed?
- Could helpers be merged into other files?
- Too much documentation?

---

## 6. core/coordinator/ Directory

### Files to Review:
- [ ] `__init__.py` - Coordinator exports
- [ ] `composable_workflow_manager.py` - Composable workflow management
- [ ] `coordinator.py` - Main coordinator implementation
- [ ] `infrastructure.py` - Infrastructure setup
- [ ] `phase_management.py` - Workflow phase management
- [ ] `protocols.py` - Coordinator protocols
- [ ] `yaml_interpreter.py` - YAML workflow interpretation
- [ ] `README.md` - Documentation

### Subdirectory: coordinator/workflows/
- [ ] `__init__.py`
- [ ] `backtest.py`
- [ ] `container_factories.py`
- [ ] `containers_pipeline.py`
- [ ] `feature_hub_workflow.py`
- [ ] `multi_strategy_aggregation.py`
- [ ] `multi_strategy_config.py`
- [ ] `optimization_workflows.py`
- [ ] `signal_replay.py`
- [ ] `strategy_coordinator.py`
- [ ] `walk_forward_workflow.py`
- [ ] `README.md`

### Subdirectory: coordinator/workflows/modes/
- [ ] `__init__.py`
- [ ] `backtest.py`
- [ ] `signal_generation.py`
- [ ] `signal_replay.py`

### Key Questions:
- Too many workflow files?
- Is phase_management.py necessary?
- Could workflows be simplified/consolidated?
- Duplicate backtest.py files?

---

## 7. core/bootstrap/ Directory

### Files to Review:
- [ ] `__init__.py` - Bootstrap exports
- [ ] `system.py` - System initialization

### Key Questions:
- Is this module necessary or could initialization be part of coordinator?
- What exactly does system.py do?

---

## 8. core/config/ Directory

### Files to Review:
- [ ] `__init__.py` - Config exports
- [ ] `example_usage.py` - Usage examples
- [ ] `schema_validator.py` - Schema validation
- [ ] `schemas.py` - Configuration schemas
- [ ] `validator_integration.py` - Validator integration

### Key Questions:
- Is schema validation being used?
- Should example_usage.py be in docs instead?
- Could this be simplified to just schemas.py?

---

## 9. Other Directories to Check

### core/data_mining/
- [ ] Should this be under core/ or elsewhere?
- [ ] What's its purpose?

### Files in core/:
- [ ] `README.md` - Core module documentation
- [ ] `TYPE_IMPROVEMENTS_SUMMARY.md` - Type improvement notes

---

## Summary Actions

After review, we should:
1. **Remove unused files**
2. **Consolidate similar files**
3. **Move misplaced files**
4. **Simplify over-engineered components**
5. **Update imports and dependencies**

## Priority Issues to Address:
- [ ] Event tracing subdirectory complexity
- [ ] Too many workflow files
- [ ] Potential duplicate functionality
- [ ] Documentation files mixed with code