# Event Communication Adapter Implementation Checklist

## Problem Statement & Motivation

### Current Issue: Circular Dependencies in Event Routing
The ADMF-PC system is experiencing circular dependency issues with event routing during multi-strategy backtests. The current implementation has containers using external event routing for communication, causing infinite loops and preventing proper execution.

**Symptoms:**
- Event cycle warnings during backtest runs
- Duplicate trades being generated
- Negative cash balance issues
- Signal rejections due to failed risk checks
- 0 bars processed despite configuration

### Root Cause: Conflated Concerns
The system currently conflates two orthogonal architectural concerns:
1. **Container Hierarchy** - Configuration inheritance and logical grouping
2. **Event Flow** - Actual trading pipeline communication

This leads to rigid communication patterns that can't adapt to different research phases or organizational needs.

### Why Not Just Fix the Router?
While we could implement a simple fix (BacktestContainer routing), the system requirements demand more flexibility:

1. **Multi-Phase Research Workflows** - Different phases need different communication patterns
2. **Dynamic Performance-Based Routing** - Route signals based on strategy performance
3. **Future Distributed Deployment** - Scale to thousands of containers across machines
4. **A/B Testing Communication Patterns** - Test if communication affects performance
5. **Complex Ensemble Routing** - Route based on regime, confidence, and market conditions

## Solution: Event Communication Adapters

### Core Concept
Separate container organization from event communication patterns using pluggable adapters. This allows:
- Same container hierarchy with different communication patterns
- Easy reconfiguration without code changes
- Support for complex routing scenarios
- Future-proof distributed deployment

### Adapter Types
1. **Pipeline** - Linear flow (Data → Indicators → Strategies → Risk → Execution)
2. **Hierarchical** - Parent-child broadcast with aggregation
3. **Broadcast** - One-to-many distribution
4. **Selective** - Rule-based routing

## Implementation Checklist

### Phase 1: Foundation (Week 1-2) ✓

#### Step 1: Create Base Adapter Interface
- [ ] Create `src/core/communication/` directory
- [ ] Implement `base_adapter.py` with:
  - [ ] Abstract `CommunicationAdapter` class
  - [ ] Logging integration via `ContainerLogger`
  - [ ] Metrics tracking (events processed, errors, latency)
  - [ ] Lifecycle methods (setup, cleanup)
  - [ ] Correlation ID support for event tracking

#### Step 2: Implement Pipeline Adapter (Priority 1)
- [ ] Create `pipeline_adapter.py` with:
  - [ ] Linear pipeline setup from configuration
  - [ ] Event transformation support via `EventTransformer`
  - [ ] Stage-by-stage wiring with logging
  - [ ] Latency tracking per stage
  - [ ] Error handling and recovery

#### Step 3: Create Communication Factory
- [ ] Implement `factory.py` with:
  - [ ] `EventCommunicationFactory` class
  - [ ] Adapter registry for type lookup
  - [ ] `CommunicationLayer` management class
  - [ ] Configuration-driven adapter creation
  - [ ] Lifecycle management for all adapters

### Phase 2: Coordinator Integration (Week 2-3)

#### Step 4: Integrate with WorkflowCoordinator
- [ ] Modify `src/core/coordinator/coordinator.py`:
  - [ ] Add `communication_factory` initialization
  - [ ] Implement `setup_communication()` method
  - [ ] Enhance `get_system_status()` with communication metrics
  - [ ] Update `shutdown()` to cleanup communication layer
  - [ ] Ensure logging integration throughout

#### Step 5: Create Configuration Examples
- [ ] Create example configurations:
  - [ ] Simple pipeline for basic workflows
  - [ ] Strategy-first organizational pattern
  - [ ] Classifier-first pattern (prep for hierarchical)
  - [ ] Multi-phase research configurations

#### Step 6: Fix Current Backtest Issues
- [ ] Update container implementations to use adapters
- [ ] Remove circular external event configurations
- [ ] Test multi-strategy backtest with pipeline adapter
- [ ] Verify no more event cycle warnings

### Phase 3: Additional Adapters (Week 3-4)

#### Step 7: Implement Broadcast Adapter
- [ ] Create `broadcast_adapter.py` with:
  - [ ] One-to-many event distribution
  - [ ] Event cloning for isolation
  - [ ] Success/failure tracking per target
  - [ ] Configurable delivery guarantees

#### Step 8: Implement Hierarchical Adapter
- [ ] Create `hierarchical_adapter.py` with:
  - [ ] Parent-to-children broadcast
  - [ ] Children-to-parent aggregation
  - [ ] Regime context propagation
  - [ ] Performance rollup support

#### Step 9: Implement Selective Adapter
- [ ] Create `selective_adapter.py` with:
  - [ ] Rule-based routing engine
  - [ ] Condition evaluation with metrics
  - [ ] Dynamic rule configuration
  - [ ] Performance tracking per rule

### Phase 4: Testing & Validation (Week 3-4)

#### Step 10: Unit Tests
- [ ] Test adapter lifecycle (creation, setup, cleanup)
- [ ] Test pipeline event flow
- [ ] Test broadcast delivery guarantees
- [ ] Test selective rule evaluation
- [ ] Test error handling and recovery

#### Step 11: Integration Tests
- [ ] Test full coordinator integration
- [ ] Test multi-adapter configurations
- [ ] Test adapter switching at runtime
- [ ] Test communication metrics collection
- [ ] Test logging correlation across adapters

#### Step 12: Performance Tests
- [ ] Benchmark adapter overhead
- [ ] Test high-frequency event scenarios
- [ ] Validate latency targets:
  - [ ] Pipeline: < 10ms end-to-end
  - [ ] Broadcast: < 5ms to all targets
  - [ ] Selective: < 2ms rule evaluation

### Phase 5: Production Deployment (Week 4+)

#### Step 13: Production Configuration
- [ ] Create production adapter configurations
- [ ] Setup monitoring integration
- [ ] Configure alerting thresholds
- [ ] Document operational procedures

#### Step 14: Migration from Current System
- [ ] Create migration guide
- [ ] Update existing configurations
- [ ] Plan phased rollout
- [ ] Setup rollback procedures

#### Step 15: Monitoring & Operations
- [ ] Implement dashboard integration
- [ ] Setup performance monitoring
- [ ] Create runbooks for common issues
- [ ] Establish SLAs for communication patterns

## Key Implementation Details

### Event Correlation
Every event flowing through adapters must have a correlation ID for tracking:
```python
correlation_id = event.get_correlation_id() or f"{adapter_type}_{uuid.uuid4().hex[:8]}"
event.set_correlation_id(correlation_id)
```

### Logging Integration
All adapters use the existing container-aware logging system:
```python
self.logger = ContainerLogger(
    coordinator_id,
    f"adapter_{adapter_type}",
    base_log_dir=str(log_manager.base_log_dir)
)
```

### Configuration Schema
```yaml
communication:
  adapters:
    - type: "pipeline"
      name: "main_flow"
      containers: ["data", "indicators", "strategies", "risk", "execution"]
    - type: "broadcast"
      source: "indicators"
      targets: ["strategy_001", "strategy_002", "strategy_003"]
```

### Metrics Collection
Each adapter tracks:
- Events processed
- Error count and rate
- Average latency
- Events per second
- Uptime

## Success Criteria

1. **Immediate Goals**
   - [ ] Multi-strategy backtest runs without circular dependencies
   - [ ] All events properly routed with correlation tracking
   - [ ] Full logging integration with existing system

2. **Architecture Goals**
   - [ ] Clean separation of container organization from communication
   - [ ] Easy to switch communication patterns via configuration
   - [ ] Support for complex routing scenarios

3. **Performance Goals**
   - [ ] No performance degradation vs current system
   - [ ] Support for 1000+ containers
   - [ ] Sub-10ms latency for standard flows

4. **Operational Goals**
   - [ ] Full observability via logging
   - [ ] Integrated metrics and monitoring
   - [ ] Graceful degradation and error recovery

## Getting Started

1. **First Task**: Implement Phase 1, Steps 1-3 (Base adapter, Pipeline adapter, Factory)
2. **Quick Win**: Test pipeline adapter with simple 3-container flow
3. **Validation**: Run problematic multi-strategy backtest with new adapter
4. **Next Steps**: Integrate with coordinator and expand adapter types

The pipeline adapter alone will solve the immediate circular dependency problem while laying the foundation for more sophisticated communication patterns needed for advanced research workflows and distributed deployment.