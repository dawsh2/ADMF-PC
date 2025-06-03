# Coordinator Modes

Complete reference for ADMF-PC's coordinator execution modes, container patterns, and YAML-driven workflow orchestration based on the actual implementation.

## 🎯 Coordinator Overview

ADMF-PC provides two coordinator implementations for workflow orchestration:

- **Base Coordinator**: Core workflow execution with lazy loading and plugin architecture
- **YAML Coordinator**: Zero-code workflow execution driven by YAML configuration files

The coordinator orchestrates workflows through container composition, event routing, and phase management while maintaining complete isolation between execution environments.

### Core Implementation Features

**From actual source code**:
- **Lazy Loading**: Complex dependencies loaded only when needed to avoid import cycles
- **Plugin Architecture**: Clean separation of concerns with minimal imports
- **Shared Services Registry**: Centralized service management with versioning
- **Container Composition Engine**: Pre-built patterns for common workflow types
- **Event Router**: Cross-container communication with scoped routing
- **Phase Management**: Workflow checkpointing and resumability for large optimizations

## 🔧 Execution Modes

The coordinator implements four execution modes defined in the `ExecutionMode` enum:

### AUTO Mode

**Description**: Coordinator automatically chooses the best execution mode based on workflow characteristics

```yaml
coordinator:
  execution_mode: "AUTO"
```

**How it works**:
- Analyzes workflow configuration to determine optimal execution strategy
- Falls back gracefully if composable containers aren't available
- Chooses between TRADITIONAL, COMPOSABLE, or HYBRID based on:
  - Workflow complexity
  - Available system resources
  - Container pattern requirements

**Use cases**:
- Default mode for most users
- Production deployments where optimal performance is desired
- When unsure which mode to use

### TRADITIONAL Mode

**Description**: Uses traditional workflow managers for backward compatibility

```yaml
coordinator:
  execution_mode: "TRADITIONAL"
```

**How it works**:
- Uses existing workflow managers: `BacktestManager`, `OptimizationManager`, etc.
- Single-threaded sequential execution
- Direct component instantiation without containers
- Simpler execution path with minimal overhead

**Use cases**:
- Simple backtests with minimal resource requirements
- Development and debugging
- Fallback when composable containers are not available
- Legacy workflow compatibility

**Performance characteristics**:
- Lower memory usage (50-200MB typical)
- Single CPU core utilization
- Baseline execution speed reference

### COMPOSABLE Mode

**Description**: Uses the new Protocol + Composition architecture with container patterns

```yaml
coordinator:
  execution_mode: "COMPOSABLE"
```

**How it works**:
- Creates containers based on predefined patterns
- Uses event routing for cross-container communication
- Complete isolation between containers
- Supports parallel execution and advanced patterns

**Container patterns available**:
- `full_backtest`: Complete workflow with all components
- `simple_backtest`: Simplified pattern with automatic inference
- `signal_generation`: Signal capture for later replay
- `signal_replay`: Fast optimization using captured signals

**Use cases**:
- Complex multi-component workflows
- Parallel optimization scenarios
- Advanced research requiring container isolation
- Signal capture and replay optimization

**Performance characteristics**:
- Higher memory usage but better parallelization
- Supports multiple CPU cores
- 10-100x faster for optimization via signal replay

### HYBRID Mode

**Description**: Mix of TRADITIONAL and COMPOSABLE modes for different workflow phases

```yaml
coordinator:
  execution_mode: "HYBRID"
```

**How it works**:
- Different phases can use different execution modes
- Automatically transitions between modes based on phase requirements
- Optimizes resource usage across complex workflows

**Example hybrid configuration**:
```yaml
workflow:
  type: "optimization"
  phases:
    - name: "coarse_optimization"
      execution_mode: "COMPOSABLE"
      container_pattern: "signal_replay"  # Fast signal replay
      
    - name: "fine_optimization"
      execution_mode: "COMPOSABLE" 
      container_pattern: "full_backtest"  # Detailed backtesting
      
    - name: "validation"
      execution_mode: "TRADITIONAL"      # Simple validation
```

**Use cases**:
- Multi-phase optimization workflows
- Resource-optimized execution
- Complex research projects with varying requirements

## 📋 Container Patterns

The Container Composition Engine provides pre-built patterns accessible via YAML configuration:

### Full Backtest Pattern

**Pattern name**: `full_backtest`

**Description**: Complete backtest workflow with all components in hierarchy

```yaml
container_pattern: "full_backtest"
```

**Container structure**:
```
Data Container (root)
├── Indicator Container
│   └── Classifier Container
│       └── Risk Container
│           └── Portfolio Container
│               └── Strategy Container
└── Execution Container (peer)
```

**Event flow**:
```
Data → Indicators → Classifier → Risk → Portfolio → Strategy → Execution
```

**Required capabilities**: `data.historical`, `execution.backtest`

**Use cases**:
- Complete strategy backtesting
- Research requiring all components
- Full workflow validation

### Simple Backtest Pattern

**Pattern name**: `simple_backtest`

**Description**: Simplified pattern with automatic indicator inference

```yaml
container_pattern: "simple_backtest"
```

**Container structure**:
```
Backtest Container (root)
├── Data Container (peer)
├── Indicator Container (peer)  
├── Classifier Container (peer)
│   ├── Risk Container (child)
│   ├── Portfolio Container (child)
│   └── Strategy Container (child)
└── Execution Container (peer)
```

**Automatic inference**:
- System automatically determines required indicators from strategy configuration
- Creates minimal necessary indicator calculations
- Optimizes for fast execution

**Use cases**:
- Quick strategy testing
- Simple backtests
- Automated workflow construction

### Signal Generation Pattern

**Pattern name**: `signal_generation`

**Description**: Generate and capture trading signals without execution

```yaml
container_pattern: "signal_generation"
```

**Container structure**:
```
Data Container (root)
├── Indicator Container
├── Classifier Container
├── Strategy Container
└── Analysis Container (signal storage)
```

**Signal capture**:
- Captures all trading signals with metadata
- Stores signals for later replay
- Enables signal analysis and ensemble methods

**Use cases**:
- Signal research and analysis
- Building signal databases
- Ensemble strategy development

### Signal Replay Pattern

**Pattern name**: `signal_replay`

**Description**: Fast optimization using pre-captured signals (10-100x speedup)

```yaml
container_pattern: "signal_replay"
```

**Container structure**:
```
Signal Log Container (root)
├── Ensemble Container
├── Risk Container
├── Portfolio Container
└── Execution Container
```

**Performance advantage**:
- Skips data loading and indicator calculation
- Focuses on risk management and execution optimization
- Enables large-scale parameter optimization

**Use cases**:
- Parameter optimization
- Risk management tuning
- Ensemble strategy optimization

## 🎼 YAML Configuration

### Workflow Configuration

The YAML Coordinator enables zero-code workflow execution:

```yaml
# Basic workflow structure
name: "My Trading Strategy"
type: "backtest"  # backtest|optimization|live_trading|analysis
description: "Optional description"

# Execution configuration
coordinator:
  execution_mode: "AUTO"  # AUTO|TRADITIONAL|COMPOSABLE|HYBRID
  container_pattern: "simple_backtest"  # Optional override

# Data configuration
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  max_bars: 1000  # Optional limit for testing

# Strategy configuration (zero-code!)
strategies:
  - name: "momentum_strategy"
    type: "momentum"
    allocation: 1.0
    parameters:
      fast_period: 10
      slow_period: 20
      signal_threshold: 0.01

# Portfolio configuration
portfolio:
  initial_capital: 100000
  currency: "USD"
  commission:
    type: "fixed"
    value: 1.0

# Risk management
risk:
  position_sizers:
    - type: "fixed_percentage"
      parameters:
        position_size_pct: 0.02
  limits:
    - type: "max_position_value"
      max_value: 10000
```

### Multi-Phase Workflows

For complex optimization workflows:

```yaml
name: "Multi-Phase Optimization"
type: "optimization"

# Phase definitions
phases:
  - name: "coarse_grid_search"
    type: "optimization"
    container_pattern: "signal_replay"  # Fast optimization
    config:
      optimization:
        method: "grid"
        parameters:
          fast_period: [5, 10, 15, 20]
          slow_period: [20, 30, 40, 50]
    
  - name: "fine_bayesian_optimization"
    type: "optimization"
    container_pattern: "full_backtest"  # Detailed backtesting
    inputs: ["coarse_grid_search.top_10_results"]
    config:
      optimization:
        method: "bayesian"
        n_trials: 100
        
  - name: "walk_forward_validation"
    type: "validation"
    execution_mode: "TRADITIONAL"  # Simple validation
    inputs: ["fine_bayesian_optimization.best_parameters"]
```

### Container Pattern Overrides

You can customize container patterns via YAML:

```yaml
# Override default pattern behavior
container_overrides:
  pattern: "full_backtest"
  modifications:
    # Add custom container
    custom_containers:
      - name: "custom_analyzer"
        role: "ANALYSIS"
        parent: "strategy"
        
    # Modify event routing
    event_routing:
      - source: "strategy"
        target: "custom_analyzer"
        event_types: ["TradingSignal"]
```

## 🔄 Phase Management

The phase management system supports complex multi-phase workflows with checkpointing and resumability:

### Phase Execution

```yaml
# Phase configuration
phase_management:
  checkpointing:
    enabled: true
    checkpoint_frequency: "every_100_trials"
    checkpoint_path: "checkpoints/"
    
  result_streaming:
    enabled: true
    stream_top_n: 50  # Keep top 50 results in memory
    disk_storage: true
    
  container_naming:
    strategy: "{phase}_{regime}_{strategy}_{params_hash}_{timestamp}"
    consistent_ids: true  # Same strategy gets same ID across regimes
```

### Phase Transitions

```yaml
# Data flow between phases
phase_transitions:
  - from_phase: "optimization"
    to_phase: "validation"
    data_transfer:
      - source: "optimization.best_parameters"
        target: "validation.strategy_config"
        
  - from_phase: "validation"
    to_phase: "live_trading"
    validation_required: true
    success_criteria:
      min_sharpe_ratio: 1.0
      max_drawdown: 0.15
```

## 🚀 Usage Examples

### Running Workflows

```bash
# Execute YAML workflow
python main.py --config config/simple_backtest.yaml

# With overrides
python main.py --config config/optimization.yaml --bars 1000 --parallel 8

# Resume from checkpoint
python main.py --config config/large_optimization.yaml --resume checkpoints/latest/
```

### Programmatic Usage

```python
from src.core.coordinator import YAMLCoordinator, ExecutionMode

# Create coordinator
coordinator = YAMLCoordinator()

# Execute workflow
result = await coordinator.execute_yaml_workflow("config/backtest.yaml")

# Check results
if result.success:
    print(f"Workflow completed: {result.data}")
else:
    print(f"Workflow failed: {result.errors}")
```

### Container Composition

```python
from src.core.containers import get_global_composition_engine

# Get composition engine
engine = get_global_composition_engine()

# Compose pattern with overrides
container = engine.compose_pattern(
    pattern_name='simple_backtest',
    config_overrides={'max_bars': 1000}
)

# Execute container
result = await container.execute()
```

## 🔍 Configuration Validation

The system provides comprehensive configuration validation:

```python
from src.core.config import ConfigSchemaValidator

# Validate configuration
validator = ConfigSchemaValidator()
result = validator.validate_config("config/my_config.yaml")

if result.is_valid:
    print("Configuration is valid")
else:
    print("Validation errors:")
    for error in result.errors:
        print(f"  - {error}")
        
    print("Warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")
```

## 🤔 Common Questions

**Q: Which execution mode should I use?**
A: Start with `AUTO` mode. It automatically chooses the best execution strategy for your workflow.

**Q: How do I enable signal replay optimization?**
A: Use `container_pattern: "signal_replay"` in your configuration. You'll need pre-captured signals from a signal generation run.

**Q: Can I create custom container patterns?**
A: Yes, through the container composition engine. See the Container Patterns section for override examples.

**Q: How does phase management work?**
A: For multi-phase workflows, the system automatically manages data flow between phases and provides checkpointing for large optimizations.

**Q: Is the system really zero-code?**
A: Yes! All strategy logic, risk management, and workflow orchestration is configured via YAML. No Python coding required.

---

Continue to [Container Patterns](container-patterns.md) for detailed container organization patterns →