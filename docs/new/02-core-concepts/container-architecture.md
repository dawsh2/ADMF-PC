# Container Architecture

Container architecture is the foundation that enables ADMF-PC's zero-code approach to scale from laptop experiments to institutional-grade systems. This document explains how complete isolation enables massive parallelization, perfect reproducibility, and zero interference.

## 🏗️ What Are Containers?

In ADMF-PC, a **container** is a completely isolated execution environment containing:

- **Event Bus**: Isolated communication system
- **State**: All mutable data and variables
- **Components**: Strategy, risk manager, data handler, etc.
- **Resources**: Memory allocation, CPU limits
- **Lifecycle**: Creation, initialization, execution, cleanup

Each container maintains internal isolation while supporting controlled communication through configurable event adapters.

## 🔒 Complete Isolation

### Isolated Event Buses

Each container has its own event bus:

```
Container A               Container B               Container C
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Event Bus A    │      │  Event Bus B    │      │  Event Bus C    │
│  ├─ BAR         │      │  ├─ SIGNAL      │      │  ├─ ORDER       │
│  ├─ INDICATOR   │      │  ├─ RISK_CHECK  │      │  ├─ FILL        │
│  └─ SIGNAL      │      │  └─ ORDER       │      │  └─ POSITION    │
└─────────────────┘      └─────────────────┘      └─────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                        Event Router (if needed)
```

**Benefits**:
- **Controlled Communication**: Cross-container events go through configurable adapters
- **Parallel Processing**: All containers can run simultaneously
- **Memory Isolation**: Each container has separate memory space
- **Failure Boundaries**: One container crash doesn't bring down others

### State Isolation

Each container maintains its own state:

```python
# Container A state (completely separate)
container_a = {
    'current_position': 1000,
    'unrealized_pnl': 2500.0,
    'last_signal_time': '2023-12-01 10:30:00',
    'strategy_parameters': {'fast_period': 10, 'slow_period': 20}
}

# Container B state (completely separate)  
container_b = {
    'current_position': -500,
    'unrealized_pnl': -1200.0,
    'last_signal_time': '2023-12-01 10:31:00',
    'strategy_parameters': {'fast_period': 5, 'slow_period': 50}
}
```

Direct state access between containers is prevented, avoiding:
- Data corruption
- Race conditions
- Unintended interactions
- Memory leaks

## 🚀 Container Lifecycle

### 1. Creation Phase
```
CREATED → Configuration loaded, container initialized
```

### 2. Initialization Phase  
```
INITIALIZING → Components created, dependencies wired, event handlers registered
```

### 3. Ready Phase
```
READY → All components initialized, waiting for start signal
```

### 4. Running Phase
```
RUNNING → Processing events, executing strategy logic
```

### 5. Disposal Phase
```
DISPOSING → Cleanup resources, save final state, terminate
```

### Lifecycle Management

```yaml
# Container lifecycle configuration
containers:
  momentum_strategy:
    lifecycle:
      initialization_timeout: 30s
      graceful_shutdown_timeout: 10s
      restart_policy: "on_failure"
      max_restarts: 3
      
    resources:
      memory_limit: "512MB"
      cpu_limit: "0.5"
      
    monitoring:
      health_check_interval: 5s
      metrics_collection: true
```

## 📦 Container Types

ADMF-PC uses specialized container types optimized for different use cases:

### Full Backtest Container
**Purpose**: Complete strategy execution with all components
```yaml
container_type: "full_backtest"
components:
  - data_handler
  - indicator_calculator  
  - strategy_engine
  - risk_manager
  - execution_engine
  - portfolio_tracker
```

**Performance**: Baseline speed, complete functionality
**Use Cases**: Strategy validation, detailed analysis, production simulation

### Signal Replay Container
**Purpose**: 10-100x faster optimization using pre-generated signals
```yaml
container_type: "signal_replay"
components:
  - signal_loader        # Loads pre-generated signals
  - risk_manager        # Re-applies risk rules
  - execution_engine    # Simulates execution
  - portfolio_tracker   # Tracks performance
```

**Performance**: 10-100x faster than full backtest
**Use Cases**: Parameter optimization, ensemble weight optimization

### Signal Generation Container  
**Purpose**: Pure signal capture for later replay
```yaml
container_type: "signal_generation"
components:
  - data_handler
  - indicator_calculator
  - strategy_engine
  - signal_exporter     # Saves signals to file
```

**Performance**: 2-3x faster than full backtest
**Use Cases**: Signal analysis, strategy research, replay preparation

### Analysis Container
**Purpose**: Statistical analysis without trading simulation
```yaml
container_type: "analysis"
components:
  - data_loader
  - statistical_analyzer
  - report_generator
```

**Performance**: Very fast, minimal memory
**Use Cases**: Regime detection, correlation analysis, feature engineering

## 🔄 Container Communication

### Cross-Container Communication

When containers need to communicate, they use **Configurable Event Adapters**:

```
Container A                 Event Adapter                    Container B
┌─────────────────┐        ┌─────────────────┐            ┌─────────────────┐
│                 │        │                 │            │                 │
│  emit(signal)   │──────→ │ Pipeline/       │──────────→ │ receive(signal) │
│                 │        │ Broadcast/      │            │                 │
│                 │        │ Hierarchical/   │            │                 │
│                 │        │ Selective       │            │                 │
└─────────────────┘        └─────────────────┘            └─────────────────┘
```

**Key Principles**:
- **Configurable Patterns**: Pipeline, broadcast, hierarchical, selective routing
- **Event-Only Communication**: All communication through events
- **Adapter-Mediated**: Communication patterns defined by adapters
- **Configuration-Driven**: Event routing and patterns defined in YAML

### Container Organization Patterns

Containers can be organized in different patterns:

#### Strategy-First Organization
```
Data Container
    ↓
Strategy Container A ──→ Execution Container
Strategy Container B ──→ Execution Container  
Strategy Container C ──→ Execution Container
```

#### Classifier-First Organization
```
Data Container
    ↓
Classifier Container
    ├──→ Risk Container A (Conservative)
    └──→ Risk Container B (Aggressive)
            ↓
    Execution Container
```

#### Risk-First Organization
```
Data Container
    ↓
Strategy Container A ──→ Risk Container ──→ Execution Container
Strategy Container B ──→ Risk Container ──→ Execution Container
```

Each pattern uses different event adapter configurations, all managed through YAML.

## 🔧 Resource Management

### Memory Management
```yaml
containers:
  high_frequency_strategy:
    resources:
      memory_limit: "2GB"
      memory_reserved: "512MB"
      gc_frequency: "aggressive"
      
  research_container:
    resources:
      memory_limit: "8GB" 
      memory_reserved: "1GB"
      gc_frequency: "standard"
```

### CPU Allocation
```yaml
containers:
  parallel_optimization:
    resources:
      cpu_cores: 4
      cpu_priority: "high"
      
  background_analysis:
    resources:
      cpu_cores: 1
      cpu_priority: "low"
```

### Storage Management
```yaml
containers:
  data_intensive_strategy:
    storage:
      temp_dir: "/tmp/strategy_temp"
      max_disk_usage: "10GB"
      cleanup_policy: "on_exit"
```

## 📊 Container Monitoring

### Health Checks
```yaml
monitoring:
  health_checks:
    - name: "memory_usage"
      threshold: 80%
      action: "alert"
      
    - name: "event_processing_rate"
      minimum: 1000  # events/second
      action: "restart"
      
    - name: "error_rate"
      threshold: 5%
      action: "circuit_breaker"
```

### Performance Metrics
```yaml
metrics:
  collection_interval: 10s
  metrics:
    - container.memory.usage
    - container.cpu.usage
    - container.events.processed
    - container.events.failed
    - container.latency.p95
    - container.latency.p99
```

## 🎯 Benefits of Container Architecture

### 1. **Massive Parallelization**

Run thousands of containers simultaneously:

```yaml
# Parameter optimization with 1000 containers
optimization:
  method: "grid"
  parallel_containers: 1000
  parameters:
    fast_period: [5, 6, 7, ..., 25]     # 20 values
    slow_period: [20, 25, 30, ..., 100] # 16 values  
    # 20 × 16 × 3 other params = 960 combinations
```

Each container tests different parameters in complete isolation.

### 2. **Perfect Reproducibility**

Identical results across runs:
```yaml
reproducibility:
  random_seed: 42
  initialization_order: "deterministic"
  event_ordering: "timestamp_strict"
```

### 3. **Zero Interference**

Memory leaks, crashes, or bugs in one container don't affect others:
```
Container A: Memory leak    → Only Container A affected
Container B: Strategy crash → Only Container B affected  
Container C: Running fine   → Unaffected, continues normally
```

### 4. **Resource Efficiency**

Shared read-only services, isolated mutable state:
```
Shared Services (Read-Only):
├─ Market Data Feed
├─ Indicator Calculation Library
└─ Risk Model Library

Container-Specific (Isolated):
├─ Strategy State
├─ Position Tracking
└─ Performance Metrics
```

## ⚡ Performance Characteristics

### Container Creation
- **Time**: < 10ms per container
- **Memory**: 20-50MB base overhead per container
- **CPU**: Minimal until processing starts

### Scaling Efficiency
- **Up to 32 cores**: 95% efficiency
- **Up to 64 cores**: 90% efficiency  
- **Beyond 64 cores**: Depends on workload

### Memory Usage
```
Base container:           20MB
+ Strategy:              +10MB
+ Risk management:       +15MB
+ Data buffers:          +50MB (configurable)
+ Working memory:        +variable
──────────────────────────────
Typical total:           95MB per container
```

## 🛠️ Container Configuration

### Basic Container Setup
```yaml
containers:
  momentum_strategy:
    type: "full_backtest"
    components:
      data_handler:
        type: "csv_handler"
        buffer_size: 10000
        
      strategy:
        type: "momentum" 
        params:
          fast_period: 10
          slow_period: 20
          
      risk_manager:
        type: "fixed_size"
        position_size_pct: 0.02
```

### Advanced Container Configuration
```yaml
containers:
  advanced_momentum:
    type: "full_backtest"
    
    resources:
      memory_limit: "1GB"
      cpu_limit: 0.5
      
    lifecycle:
      startup_timeout: 30s
      shutdown_timeout: 10s
      restart_policy: "on_failure"
      
    monitoring:
      health_checks: true
      metrics_collection: true
      log_level: "INFO"
      
    communication:
      event_buffer_size: 1000
      max_event_rate: 10000  # events/second
      
    persistence:
      checkpoint_interval: 300s  # 5 minutes
      state_backup: true
```

## 🎓 Best Practices

### Container Design
1. **Single Responsibility**: Each container should have one clear purpose
2. **Minimal State**: Keep container state as small as possible  
3. **Stateless When Possible**: Use shared read-only services
4. **Clear Interfaces**: Define event contracts clearly

### Resource Management
1. **Set Limits**: Always define memory and CPU limits
2. **Monitor Usage**: Track resource consumption
3. **Plan Capacity**: Account for peak usage scenarios
4. **Cleanup Properly**: Ensure proper resource cleanup

### Error Handling
1. **Graceful Degradation**: Handle component failures gracefully
2. **Circuit Breakers**: Use circuit breakers for external dependencies
3. **Restart Policies**: Define appropriate restart behavior
4. **Fallback Strategies**: Have fallback plans for critical failures

## 🤔 Common Questions

**Q: How many containers can I run simultaneously?**
A: Thousands! The exact number depends on your hardware, but ADMF-PC efficiently scales to the limits of your system.

**Q: Do containers share any data?**
A: Only read-only services like market data feeds are shared. All mutable state is completely isolated.

**Q: Can containers communicate directly?**
A: No! All communication goes through configurable Event Adapters to maintain isolation while enabling flexible communication patterns.

**Q: What happens if a container crashes?**
A: Only that container is affected. Other containers continue running normally. The crashed container can be restarted automatically.

## 🎯 Key Takeaways

1. **Controlled Isolation**: Internal isolation with configurable communication
2. **Parallel by Design**: Natural parallelization without race conditions
3. **Resource Controlled**: Memory, CPU, and lifecycle management
4. **Failure Isolated**: Container failures don't cascade
5. **Configuration Driven**: All behavior and communication patterns specified in YAML

Container architecture is what enables ADMF-PC to scale from simple backtests to institutional-grade trading systems while maintaining the zero-code philosophy.

---

Next: [Event-Driven Design](event-driven-design.md) - How containers communicate →