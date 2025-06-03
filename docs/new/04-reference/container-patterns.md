# Container Patterns

Complete reference for ADMF-PC's container organization patterns, event communication, and composition engine based on the actual implementation.

## 🏗️ Container Architecture Overview

ADMF-PC uses a sophisticated Container Composition Engine that organizes trading system components into isolated, composable containers. Each container maintains complete internal isolation while supporting controlled communication through an event router.

### Core Container Concepts

**Container Roles** (from actual `ContainerRole` enum):
- `BACKTEST` - Root backtest orchestration
- `DATA` - Data loading and streaming
- `INDICATOR` - Technical indicator computation
- `CLASSIFIER` - Regime classification and pattern detection
- `RISK` - Risk management and position sizing
- `PORTFOLIO` - Portfolio state tracking and optimization
- `STRATEGY` - Trading strategy execution
- `EXECUTION` - Order execution and market simulation
- `ANALYSIS` - Performance analysis and reporting
- `SIGNAL_LOG` - Signal capture and replay
- `ENSEMBLE` - Strategy ensemble management

**Container Properties**:
- **Complete Isolation**: Each container has independent state and event bus
- **Protocol-Based Communication**: Containers communicate only through events
- **Composable Architecture**: Containers can be combined into complex patterns
- **Resource Management**: Each container has allocated CPU, memory, and storage limits
- **Lifecycle Management**: Automatic initialization, execution, and cleanup

## 📋 Pre-Built Container Patterns

The Container Composition Engine provides four pre-built patterns accessible via YAML configuration:

### Full Backtest Pattern

**Pattern Name**: `full_backtest`

**Description**: Complete hierarchical backtest workflow with all components

**YAML Configuration**:
```yaml
container_pattern: "full_backtest"
```

**Container Hierarchy**:
```
Data Container (root)
├── Indicator Container
│   └── Classifier Container
│       └── Risk Container
│           └── Portfolio Container
│               └── Strategy Container
└── Execution Container (peer to root)
```

**Event Flow**:
```
Data Events → Indicators → Classification → Risk Assessment → Portfolio Updates → Strategy Decisions → Execution Orders
```

**Required Capabilities**: 
- `data.historical` - Historical data loading
- `execution.backtest` - Backtest execution simulation

**Resource Characteristics**:
- Memory: 200-500MB depending on data size
- CPU: Utilizes multiple cores for parallel indicator calculation
- Storage: Temporary files for large datasets

**Use Cases**:
- Complete strategy research and validation
- Full-featured backtesting with all components
- Academic research requiring component isolation
- Complex multi-component strategy development

### Simple Backtest Pattern

**Pattern Name**: `simple_backtest`

**Description**: Streamlined pattern with automatic indicator inference and peer container organization

**YAML Configuration**:
```yaml
container_pattern: "simple_backtest"
```

**Container Hierarchy**:
```
Backtest Container (root orchestrator)
├── Data Container (peer)
├── Indicator Container (peer, auto-inferred)
├── Classifier Container (peer)
│   ├── Risk Container (child)
│   ├── Portfolio Container (child)
│   └── Strategy Container (child)
└── Execution Container (peer)
```

**Automatic Inference Features**:
- **Indicator Auto-Detection**: System analyzes strategy configuration and automatically creates required indicators
- **Minimal Indicator Set**: Only creates indicators actually needed by the strategy
- **Optimized Execution**: Streamlined data flow for faster execution

**Example Auto-Inference**:
```yaml
# Strategy configuration
strategies:
  - type: "momentum"
    parameters:
      fast_period: 10
      slow_period: 20

# System automatically infers and creates:
# - SMA(10) indicator
# - SMA(20) indicator
# - Momentum crossover logic
```

**Use Cases**:
- Quick strategy prototyping and testing
- Educational examples and tutorials
- Simple single-strategy backtests
- Automated workflow construction

### Signal Generation Pattern

**Pattern Name**: `signal_generation`

**Description**: Captures trading signals for analysis and later replay optimization

**YAML Configuration**:
```yaml
container_pattern: "signal_generation"
```

**Container Hierarchy**:
```
Data Container (root)
├── Indicator Container
├── Classifier Container
├── Strategy Container
└── Analysis Container (signal capture)
```

**Signal Capture Process**:
1. **Signal Generation**: Strategies generate trading signals with full metadata
2. **Signal Storage**: Analysis container captures signals with timestamp, strength, confidence
3. **Metadata Enrichment**: Adds regime context, indicator values, and market conditions
4. **Compression**: Compresses signals for efficient storage and later replay

**Captured Signal Schema**:
```python
@dataclass
class CapturedSignal:
    timestamp: datetime
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    strength: float
    confidence: float
    strategy_id: str
    regime_context: Dict[str, Any]
    indicator_values: Dict[str, float]
    market_conditions: Dict[str, Any]
```

**Use Cases**:
- Building signal databases for research
- Signal quality analysis and validation
- Ensemble strategy development
- Creating datasets for machine learning

### Signal Replay Pattern

**Pattern Name**: `signal_replay`

**Description**: High-speed optimization using pre-captured signals (10-100x faster)

**YAML Configuration**:
```yaml
container_pattern: "signal_replay"
```

**Container Hierarchy**:
```
Signal Log Container (root)
├── Ensemble Container (signal processing)
├── Risk Container (risk management)
├── Portfolio Container (portfolio tracking)
└── Execution Container (order simulation)
```

**Performance Optimization**:
- **Skip Data Loading**: No need to reload market data
- **Skip Indicator Calculation**: Indicators already computed and stored with signals
- **Focus on Risk/Execution**: Optimizes only risk management and execution parameters
- **Parallel Execution**: Multiple parameter combinations tested simultaneously

**Speed Comparison**:
```
Traditional Backtest: 100% baseline speed
Signal Replay:       10-100x faster (depends on strategy complexity)

Example timing:
- 1000 parameter combinations
- Traditional: 10 hours
- Signal Replay: 6-60 minutes
```

**Use Cases**:
- Large-scale parameter optimization
- Risk management parameter tuning
- Execution algorithm optimization
- Walk-forward analysis with many periods

## 🔄 Event Communication System

ADMF-PC implements a sophisticated event router for cross-container communication:

### Event Scopes

**From actual `EventScope` enum**:
- `LOCAL` - Within the same container
- `PARENT` - To parent container
- `CHILDREN` - To all child containers
- `SIBLINGS` - To sibling containers
- `UPWARD` - Up the container hierarchy
- `DOWNWARD` - Down the container hierarchy
- `GLOBAL` - To all containers in the system

### Event Router Configuration

```yaml
# Event routing configuration
event_routing:
  # Default routing patterns
  default_patterns:
    market_data:
      scope: "DOWNWARD"
      event_types: ["BarEvent", "TickEvent"]
      
    trading_signals:
      scope: "UPWARD"
      event_types: ["TradingSignal"]
      
    portfolio_updates:
      scope: "GLOBAL"
      event_types: ["PortfolioUpdate", "PositionUpdate"]
      
  # Custom routing rules
  custom_rules:
    - source_role: "STRATEGY"
      target_role: "RISK"
      event_types: ["TradingSignal"]
      scope: "PARENT"
      
    - source_role: "DATA"
      target_role: "INDICATOR"
      event_types: ["BarEvent"]
      scope: "CHILDREN"
```

### Cross-Container Communication

**HybridContainerInterface** bridges internal container events with external routing:

```python
# Example event publication
container.publish_event(
    event=TradingSignal(symbol="SPY", action="BUY", strength=0.8),
    scope=EventScope.UPWARD,
    target_roles=[ContainerRole.RISK, ContainerRole.PORTFOLIO]
)

# Example event subscription
container.subscribe_to_events(
    event_types=[BarEvent, IndicatorEvent],
    scope=EventScope.DOWNWARD,
    handler=self.handle_market_data
)
```

## 🔧 Container Composition Engine

The composition engine manages container creation, initialization, and lifecycle:

### Pattern Registration

```python
# From actual source code - pattern registration
pattern = ContainerPattern(
    name="custom_pattern",
    description="Custom trading pattern",
    structure={
        "root": {
            "role": "data",
            "children": {
                "strategy": {"role": "strategy"},
                "execution": {"role": "execution"}
            }
        }
    },
    required_capabilities={"data.historical", "execution.backtest"}
)

# Register pattern
engine = get_global_composition_engine()
engine.register_pattern(pattern)
```

### Container Factory Registration

```python
# Register custom container type
from src.core.containers import ContainerRegistry

registry = ContainerRegistry()
registry.register_container_type(
    role=ContainerRole.CUSTOM_ANALYZER,
    factory_func=create_custom_analyzer_container,
    capabilities={"analysis.advanced", "reporting.custom"}
)
```

### Dynamic Pattern Composition

```yaml
# YAML-driven dynamic composition
container_composition:
  pattern: "custom"
  structure:
    root:
      role: "BACKTEST"
      children:
        data_loader:
          role: "DATA"
          config:
            source: "csv"
            file_path: "data/SPY_1m.csv"
            
        indicator_hub:
          role: "INDICATOR"
          config:
            indicators: ["SMA", "RSI", "MACD"]
            
        strategy_engine:
          role: "STRATEGY"
          config:
            strategy_type: "momentum"
            parameters:
              fast_period: 10
              slow_period: 20
              
  event_routing:
    - source: "data_loader"
      target: "indicator_hub"
      events: ["BarEvent"]
      
    - source: "indicator_hub"
      target: "strategy_engine"
      events: ["IndicatorEvent"]
```

## 💾 Resource Management

### Container Resource Allocation

```yaml
# Resource configuration per container
resource_management:
  container_limits:
    default:
      memory_mb: 100
      cpu_cores: 0.1
      storage_mb: 50
      
    by_role:
      DATA:
        memory_mb: 300  # Higher for data loading
        cpu_cores: 0.2
        storage_mb: 500
        
      INDICATOR:
        memory_mb: 200  # Moderate for calculations
        cpu_cores: 0.3  # Higher CPU for computations
        storage_mb: 100
        
      STRATEGY:
        memory_mb: 150  # Moderate memory needs
        cpu_cores: 0.2
        storage_mb: 50
        
      EXECUTION:
        memory_mb: 100  # Minimal memory needs
        cpu_cores: 0.1
        storage_mb: 50
        
  # Resource monitoring
  monitoring:
    enabled: true
    check_interval_seconds: 30
    alert_thresholds:
      memory_usage: 0.9
      cpu_usage: 0.95
      storage_usage: 0.85
```

### Adaptive Resource Allocation

```yaml
# Adaptive resource management
adaptive_resources:
  enabled: true
  
  # Scale resources based on workload
  scaling_rules:
    memory_scaling:
      scale_up_threshold: 0.8
      scale_down_threshold: 0.4
      min_allocation_mb: 50
      max_allocation_mb: 1000
      
    cpu_scaling:
      scale_up_threshold: 0.85
      scale_down_threshold: 0.3
      min_allocation: 0.05
      max_allocation: 2.0
      
  # Rebalancing
  rebalancing:
    frequency_seconds: 60
    algorithm: "least_loaded"
    consider_container_role: true
```

## 🔄 Container Lifecycle Management

### Initialization Process

**From actual source code lifecycle**:

1. **Resource Allocation**: CPU, memory, storage allocated per container limits
2. **Event Bus Setup**: Internal event bus created and configured
3. **Component Loading**: Strategy, risk manager, etc. loaded based on configuration
4. **Event Handler Registration**: Handlers registered for expected event types
5. **Health Check**: Container validates all components are ready
6. **Router Connection**: Container connects to cross-container event router

```python
# Container initialization sequence
async def initialize_container(
    role: ContainerRole,
    config: Dict[str, Any],
    parent: Optional[Container] = None
) -> Container:
    
    # 1. Allocate resources
    resources = allocate_container_resources(role, config)
    
    # 2. Create container
    container = create_container_instance(role, resources)
    
    # 3. Setup event bus
    await container.setup_event_bus()
    
    # 4. Load components
    await container.load_components(config)
    
    # 5. Register event handlers
    await container.register_event_handlers()
    
    # 6. Health check
    health_result = await container.health_check()
    if not health_result.healthy:
        raise ContainerInitializationError(health_result.errors)
    
    # 7. Connect to router
    await container.connect_to_router(parent)
    
    return container
```

### Container Disposal

```python
# Graceful container disposal
async def dispose_container(container: Container) -> None:
    # 1. Stop accepting new events
    await container.stop_event_processing()
    
    # 2. Process remaining events (with timeout)
    await container.flush_event_queue(timeout_seconds=30)
    
    # 3. Save state if needed
    await container.save_state()
    
    # 4. Disconnect from router
    await container.disconnect_from_router()
    
    # 5. Release resources
    await container.release_resources()
    
    # 6. Cleanup temporary files
    await container.cleanup_storage()
```

## 🎯 Pattern Selection Guide

### Choosing the Right Pattern

| Use Case | Recommended Pattern | Reason |
|----------|-------------------|---------|
| Quick strategy testing | `simple_backtest` | Fast setup, automatic inference |
| Research with all components | `full_backtest` | Complete isolation, full features |
| Signal analysis | `signal_generation` | Captures signals for analysis |
| Large parameter optimization | `signal_replay` | 10-100x faster execution |
| Custom workflow | Dynamic composition | Maximum flexibility |

### Performance Comparison

```
Pattern Performance (1000 parameter combinations):

simple_backtest:    Baseline speed, moderate memory
full_backtest:      80% of baseline, higher memory usage
signal_generation:  90% of baseline, stores signals
signal_replay:      10-100x faster, minimal memory

Memory Usage:
simple_backtest:    100-300MB
full_backtest:      200-500MB  
signal_generation:  150-400MB + signal storage
signal_replay:      50-150MB
```

## 🔧 Custom Pattern Development

### Creating Custom Patterns

```python
# Define custom container pattern
from src.core.containers import ContainerPattern, ContainerRole

custom_pattern = ContainerPattern(
    name="ml_strategy_pattern",
    description="Machine learning strategy with feature engineering",
    structure={
        "root": {
            "role": "data",
            "children": {
                "feature_engineer": {
                    "role": "indicator",  # Reuse indicator role for features
                    "children": {
                        "ml_model": {
                            "role": "strategy",
                            "children": {
                                "risk_manager": {"role": "risk"},
                                "portfolio": {"role": "portfolio"}
                            }
                        }
                    }
                },
                "execution": {"role": "execution"}
            }
        }
    },
    required_capabilities={
        "data.historical",
        "execution.backtest", 
        "ml.prediction"
    },
    optional_capabilities={
        "data.alternative",
        "ml.online_learning"
    }
)

# Register pattern
engine = get_global_composition_engine()
engine.register_pattern(custom_pattern)
```

### Using Custom Patterns

```yaml
# Use custom pattern in YAML
container_pattern: "ml_strategy_pattern"

# Override default configuration
container_overrides:
  feature_engineer:
    config:
      features: ["returns", "volatility", "momentum"]
      lookback_period: 20
      
  ml_model:
    config:
      model_type: "random_forest"
      retrain_frequency: "monthly"
      features: ["sma_ratio", "rsi", "volatility"]
```

## 🤔 Common Questions

**Q: How do containers communicate?**
A: Through the event router using scoped event publication/subscription. No direct method calls between containers.

**Q: Can I create custom container patterns?**
A: Yes, either programmatically through the ContainerPattern class or via YAML container_composition configuration.

**Q: Which pattern should I use for optimization?**
A: For large optimizations, use signal_generation first to capture signals, then signal_replay for fast parameter testing.

**Q: How does automatic indicator inference work?**
A: The system analyzes strategy configurations to determine required indicators and automatically creates only the necessary ones.

**Q: Can containers run on different machines?**
A: Currently containers run on the same machine but are designed for future distributed execution.

---

Continue to [Event Adapters](event-adapters.md) for detailed event communication specifications →