# Event Adapter Migration: Complete Hunting Guide

**Purpose**: Find and replace every deprecated communication pattern  
**Target**: Zero legacy patterns remaining after migration  
**Status**: Ready for device switch and systematic migration

---

## 🎯 SEARCH PATTERNS TO HUNT DOWN

### 1. Global State Instances
```bash
# Find all global registries and singletons
grep -r "_global_" src/
grep -r "_GLOBAL_" src/
grep -r "global " src/
grep -r "_registry" src/
grep -r "_REGISTRY" src/

# Specific patterns to eliminate
grep -r "_PORTFOLIO_CONTAINER_REGISTRY" src/
grep -r "get_global_registry" src/
grep -r "_global_composition_engine" src/
```

### 2. Direct Container Method Calls
```bash
# Find direct cross-container method calls
grep -r "\.receive_event(" src/
grep -r "\._.*_directly(" src/
grep -r "container\." src/ | grep -v "self\.container"
grep -r "Container\(" src/ | grep -v "class.*Container"

# Specific problematic patterns
grep -r "_close_all_positions_directly" src/
grep -r "portfolio_container\." src/
grep -r "execution_container\." src/
```

### 3. Pipeline Communication Patterns
```bash
# Find pipeline adapter usage
grep -r "PipelineCommunicationAdapter" src/
grep -r "pipeline_adapter" src/
grep -r "setup_pipeline" src/
grep -r "on_output_event" src/
grep -r "_stage_handlers" src/
grep -r "_wire_pipeline" src/
```

### 4. Event Bus Subscriptions (Type-Based)
```bash
# Find type-based event subscriptions
grep -r "\.subscribe(" src/
grep -r "\.publish(" src/
grep -r "EventType\." src/
grep -r "event_bus\.subscribe" src/
grep -r "self\.event_bus" src/

# Specific event types to replace
grep -r "EventType\.FILL" src/
grep -r "EventType\.SYSTEM" src/
grep -r "EventType\.ORDER" src/
grep -r "EventType\.SIGNAL" src/
```

### 5. Hybrid Interface Usage
```bash
# Find hybrid interface patterns
grep -r "HybridEventInterface" src/
grep -r "TieredEventRouter" src/
grep -r "CommunicationTier" src/
grep -r "event_tier_map" src/
grep -r "_setup_external_communication" src/
```

### 6. Event Router Registration
```bash
# Find old router patterns
grep -r "_routing_table" src/
grep -r "register_publisher" src/
grep -r "register_subscriber" src/
grep -r "EventRouter(" src/
grep -r "route_event" src/
```

### 7. Hard-coded Communication Logic
```bash
# Find hard-coded routing logic
grep -r "_setup_reverse_fill" src/
grep -r "_setup_portfolio_routing" src/
grep -r "_broadcast_system_event" src/
grep -r "forward_to_actor" src/
grep -r "forward_with_isolation" src/
```

---

## 🔍 FILE-BY-FILE HUNTING CHECKLIST

### Core Communication Files (🔴 High Priority)
- [ ] `src/core/communication/pipeline_adapter.py` - **REPLACE ENTIRELY**
- [ ] `src/core/events/hybrid_interface.py` - **REPLACE ENTIRELY** 
- [ ] `src/core/events/tiered_router.py` - **MIGRATE TO SEMANTIC**
- [ ] `src/core/events/routing/router.py` - **MIGRATE TO SEMANTIC**
- [ ] `src/core/containers/composition_engine.py` - **REMOVE GLOBAL STATE**

### Container Implementation Files (🔴 High Priority)
- [ ] `src/execution/containers_pipeline.py` - **EXTENSIVE CHANGES**
  - Remove `_PORTFOLIO_CONTAINER_REGISTRY`
  - Replace `on_output_event()` patterns
  - Replace `receive_event()` direct calls
  - Migrate to semantic events
- [ ] `src/core/containers/composable.py` - **PROTOCOL UPDATES**
- [ ] `src/core/containers/bootstrap.py` - **INITIALIZATION CHANGES**

### Strategy and Risk Files (🟡 Medium Priority)
- [ ] `src/strategy/enhanced_strategy_container.py`
- [ ] `src/strategy/classifiers/enhanced_classifier_container.py`
- [ ] `src/risk/risk_container.py`
- [ ] `src/risk/risk_portfolio.py`

### Coordinator Files (🟡 Medium Priority)
- [ ] `src/core/coordinator/coordinator.py`
- [ ] `src/core/coordinator/composable_workflow_manager_pipeline.py`
- [ ] `src/core/coordinator/phase_management.py`

---

## 🚨 CRITICAL INSTANCES TO REPLACE

### Instance 1: Global Portfolio Registry
**Files**: `src/execution/containers_pipeline.py`
```python
# FIND AND DESTROY
_PORTFOLIO_CONTAINER_REGISTRY = None

# IN INIT
global _PORTFOLIO_CONTAINER_REGISTRY
_PORTFOLIO_CONTAINER_REGISTRY = self

# IN USAGE  
portfolio_container = _PORTFOLIO_CONTAINER_REGISTRY
await portfolio_container._close_all_positions_directly(reason)
```

### Instance 2: Direct Container Method Calls
**Files**: Throughout container implementations
```python
# FIND AND DESTROY
target_container.receive_event(event)
container.on_output_event(handler)
self._parent_container.child_containers
```

### Instance 3: Type-Based Event Subscriptions
**Files**: All container files
```python
# FIND AND DESTROY
self.event_bus.subscribe(EventType.FILL, handler)
self.event_bus.publish(event)
```

### Instance 4: Hard-coded Pipeline Logic
**Files**: `src/core/communication/pipeline_adapter.py`
```python
# FIND AND DESTROY
def _setup_reverse_fill_routing(self):
def _setup_portfolio_routing(self):
def _broadcast_system_event(self, event):
```

---

## 🔧 REPLACEMENT PATTERNS

### Replace Global State With Dependency Injection
```python
# OLD (DEPRECATED)
_PORTFOLIO_CONTAINER_REGISTRY = None

# NEW (SEMANTIC ADAPTER)
@inject
def __init__(self, semantic_adapter: SemanticEventAdapter):
    self.semantic_adapter = semantic_adapter
```

### Replace Direct Calls With Semantic Events  
```python
# OLD (DEPRECATED)
portfolio_container._close_all_positions_directly(reason)

# NEW (SEMANTIC ADAPTER)
await self.semantic_adapter.send_command(
    command="close_all_positions",
    target_capability="portfolio_management",
    context={"reason": reason, "urgency": "immediate"}
)
```

### Replace Type Subscriptions With Intent Subscriptions
```python
# OLD (DEPRECATED) 
self.event_bus.subscribe(EventType.FILL, handler)

# NEW (SEMANTIC ADAPTER)
await self.semantic_adapter.subscribe_to_intent(
    intent="position_update",
    context_filter={"asset_class": "equities"},
    callback=semantic_handler
)
```

### Replace Pipeline Logic With Semantic Routing
```python
# OLD (DEPRECATED)
target_container.receive_event(transformed_event)

# NEW (SEMANTIC ADAPTER)
await self.semantic_adapter.route_event(
    event=event,
    semantic_intent="data_processing",
    context={"pipeline_stage": "indicators"}
)
```

---

## 📋 VERIFICATION COMMANDS

### After Each File Migration
```bash
# Verify no deprecated patterns remain in file
grep -n "receive_event\|_REGISTRY\|event_bus\.subscribe\|on_output_event" filename.py

# Should return ZERO matches for deprecated patterns
```

### After Complete Migration
```bash
# Global verification - should return ZERO matches
grep -r "_PORTFOLIO_CONTAINER_REGISTRY" src/
grep -r "\.receive_event(" src/
grep -r "_global_registry" src/  
grep -r "on_output_event" src/
grep -r "_setup_reverse_fill" src/

# Verify new patterns are in place
grep -r "SemanticEventAdapter" src/
grep -r "semantic_adapter\." src/
grep -r "send_command\|subscribe_to_intent" src/
```

---

## 🎯 MIGRATION ORDER (Device Switch Ready)

### Step 1: Foundation Setup (Day 1)
1. Implement `SemanticEventAdapter` base classes
2. Create semantic event types (`SystemShutdownEvent`, `PortfolioEvent`, etc.)
3. Set up dependency injection framework

### Step 2: Replace Global State (Day 2-3)
1. Remove `_PORTFOLIO_CONTAINER_REGISTRY` 
2. Remove `_global_registry` and `_global_composition_engine`
3. Replace with dependency injection

### Step 3: Migrate Core Communication (Day 4-7)
1. Replace `pipeline_adapter.py` with semantic pipeline
2. Replace `hybrid_interface.py` with unified adapter
3. Update all `receive_event()` calls

### Step 4: Update All Containers (Day 8-10)
1. Replace `on_output_event()` patterns
2. Replace type-based subscriptions
3. Add semantic capability registration

### Step 5: Verification & Testing (Day 11-12)
1. Run all verification commands
2. Test position closing still works
3. Test multi-portfolio scenarios

---

## 💾 BACKUP STRATEGY

### Before Starting Migration
```bash
# Create migration branch
git checkout -b feature/semantic-event-adapters

# Tag current working state
git tag before-semantic-migration

# Backup critical files
cp src/execution/containers_pipeline.py src/execution/containers_pipeline.py.backup
cp src/core/communication/pipeline_adapter.py src/core/communication/pipeline_adapter.py.backup
```

### Test Rollback Capability
```bash
# Verify you can rollback to working state
git checkout before-semantic-migration
python main.py --config config/multi_strategy_test.yaml --bars 50
# Should show 0 positions at end

git checkout feature/semantic-event-adapters
# Continue migration
```

---

## 🚀 SUCCESS CRITERIA

### Migration Complete When:
- [ ] All verification commands return ZERO deprecated patterns
- [ ] Position closing still works (0 positions at end)
- [ ] Can run multiple portfolio instances simultaneously  
- [ ] All containers use semantic event communication only
- [ ] No global state remaining anywhere
- [ ] Clean semantic event flow traceable through logs

### Performance Validation:
- [ ] Backtest performance same or better than before
- [ ] Memory usage stable (no leaks from old patterns)
- [ ] Event latency within acceptable bounds
- [ ] Error handling and recovery works properly

---

**Ready for Device Switch**: All hunting patterns documented  
**Estimated Timeline**: 10-12 days for complete migration  
**Risk Level**: High but systematic with clear rollback plan  
**Success Measure**: Zero deprecated patterns + working position closing