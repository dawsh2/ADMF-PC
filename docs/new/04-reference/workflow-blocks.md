# Workflow Blocks

Complete reference for ADMF-PC's workflow building blocks including core blocks, parameters, data flow, and composition rules based on the actual implementation.

## 🎯 Workflow Block Overview

ADMF-PC uses composable workflow blocks that can be combined to create complex trading research workflows. Each block encapsulates a specific type of analysis or operation while maintaining clean data flow interfaces.

### Core Block Types

**From actual implementation**:
- **Backtest Block**: Single strategy backtesting with performance analysis
- **Optimization Block**: Parameter optimization using various algorithms
- **Analysis Block**: Signal generation, feature analysis, and research tools
- **Validation Block**: Out-of-sample validation and walk-forward analysis
- **Signal Generation Block**: Signal capture for later replay optimization
- **Signal Replay Block**: Fast optimization using pre-captured signals

## 📋 Core Workflow Blocks

### Backtest Block

**Block Type**: `backtest`

**Description**: Executes a single backtest with specified strategy and parameters

**YAML Configuration**:
```yaml
type: "backtest"
name: "Single Strategy Backtest"
description: "Basic momentum strategy backtest"

# Data configuration
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  max_bars: 10000  # Optional limit

# Strategy configuration
strategies:
  - type: "momentum"
    name: "momentum_strategy"
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
      max_value: 5000

# Execution configuration
coordinator:
  execution_mode: "AUTO"
  container_pattern: "simple_backtest"
```

**Input Requirements**:
- Market data (historical)
- Strategy configuration
- Portfolio settings
- Risk management rules

**Output Data**:
```python
@dataclass
class BacktestResult:
    performance_metrics: Dict[str, float]
    trade_log: List[Trade]
    portfolio_history: List[PortfolioSnapshot]
    signals: List[TradingSignal]  # Optional
    metadata: Dict[str, Any]
    
    # Performance metrics included
    performance_metrics = {
        "total_return": 0.125,
        "annualized_return": 0.089,
        "volatility": 0.156,
        "sharpe_ratio": 1.34,
        "max_drawdown": 0.087,
        "win_rate": 0.547,
        "profit_factor": 1.23
    }
```

**Use Cases**:
- Single strategy validation
- Quick strategy testing
- Performance baseline establishment
- Educational examples

### Optimization Block

**Block Type**: `optimization`

**Description**: Optimizes strategy parameters using various algorithms

**YAML Configuration**:
```yaml
type: "optimization"
name: "Parameter Optimization"
description: "Bayesian optimization of momentum strategy"

# Base configuration (same as backtest)
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  start_date: "2023-01-01"
  end_date: "2023-10-31"  # Leave 2 months for validation

strategies:
  - type: "momentum"
    optimization_target: true  # Mark for optimization
    parameters:
      fast_period: 10  # Will be optimized
      slow_period: 20  # Will be optimized
      signal_threshold: 0.01  # Will be optimized

# Optimization configuration
optimization:
  method: "bayesian"
  objective: "sharpe_ratio"
  n_trials: 200
  timeout_hours: 2
  
  # Parameter space
  parameters:
    fast_period: [5, 30]        # Continuous range
    slow_period: [20, 100]      # Continuous range
    signal_threshold: [0.001, 0.1]  # Continuous range
    
  # Optimization constraints
  constraints:
    - "fast_period < slow_period"  # Logical constraint
    
  # Performance configuration
  performance:
    parallel_workers: 8
    container_pattern: "signal_replay"  # Fast optimization
    keep_top_n: 50

# Execution configuration
coordinator:
  execution_mode: "COMPOSABLE"
  container_pattern: "signal_replay"  # 10-100x faster
```

**Available Optimization Methods**:
- `grid`: Exhaustive grid search
- `random`: Random search
- `bayesian`: Bayesian optimization (Gaussian Process)
- `genetic`: Genetic algorithm
- `differential_evolution`: Differential evolution

**Input Requirements**:
- Base strategy configuration
- Parameter space definition
- Optimization method and settings
- Objective function

**Output Data**:
```python
@dataclass
class OptimizationResult:
    best_parameters: Dict[str, Any]
    best_performance: float
    optimization_history: List[Trial]
    parameter_importance: Dict[str, float]
    convergence_plot_data: Dict[str, List[float]]
    top_n_results: List[Dict[str, Any]]
    
    # Example best result
    best_parameters = {
        "fast_period": 12,
        "slow_period": 28,
        "signal_threshold": 0.025
    }
    best_performance = 1.47  # Sharpe ratio
```

**Use Cases**:
- Parameter optimization
- Strategy improvement
- Sensitivity analysis
- Hyperparameter tuning

### Analysis Block

**Block Type**: `analysis`

**Description**: Performs various types of analysis without trading execution

**YAML Configuration**:
```yaml
type: "analysis"
name: "Signal Analysis"
description: "Analyze signal quality and characteristics"

# Data configuration
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY", "QQQ"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"

# Analysis configuration
analysis:
  type: "signal_analysis"
  
  # Signal generation
  signal_generation:
    strategies:
      - type: "momentum"
        parameters:
          fast_period: 10
          slow_period: 20
      - type: "mean_reversion"
        parameters:
          rsi_period: 14
          oversold: 30
          overbought: 70
          
  # Analysis settings
  analysis_settings:
    signal_quality_metrics: true
    correlation_analysis: true
    regime_analysis: true
    feature_importance: true
    
    # Output options
    save_signals: true
    signal_output_path: "signals/momentum_signals.parquet"
    generate_report: true

# Execution configuration
coordinator:
  execution_mode: "COMPOSABLE"
  container_pattern: "signal_generation"
```

**Analysis Types Available**:
- `signal_analysis`: Signal quality and characteristics
- `feature_analysis`: Feature importance and correlation
- `regime_analysis`: Market regime identification
- `performance_attribution`: Performance decomposition
- `risk_analysis`: Risk factor analysis

**Output Data**:
```python
@dataclass
class AnalysisResult:
    signal_metrics: Dict[str, float]
    correlation_matrix: pd.DataFrame
    feature_importance: Dict[str, float]
    regime_analysis: Dict[str, Any]
    saved_signals_path: Optional[str]
    report_path: Optional[str]
    
    # Example signal metrics
    signal_metrics = {
        "signal_count": 1247,
        "signal_frequency": 0.089,  # Signals per bar
        "average_strength": 0.342,
        "average_confidence": 0.567,
        "directional_accuracy": 0.634
    }
```

**Use Cases**:
- Signal quality research
- Feature engineering
- Strategy analysis
- Market research

### Validation Block

**Block Type**: `validation`

**Description**: Validates strategy performance using out-of-sample testing

**YAML Configuration**:
```yaml
type: "validation"
name: "Walk-Forward Validation"
description: "Validate optimized parameters"

# Input from optimization block
inputs:
  - source: "optimization_block"
    data: "best_parameters"
    
# Data configuration
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  start_date: "2023-11-01"  # Out-of-sample period
  end_date: "2023-12-31"

# Strategy configuration (parameters from optimization)
strategies:
  - type: "momentum"
    parameters: "{{ inputs.optimization_block.best_parameters }}"

# Validation configuration
validation:
  type: "walk_forward"
  
  walk_forward_settings:
    train_period_days: 63   # ~3 months
    test_period_days: 21    # ~1 month
    step_days: 7            # Weekly steps
    min_trades_required: 10
    
  # Validation criteria
  success_criteria:
    min_sharpe_ratio: 0.8
    max_drawdown: 0.15
    min_win_rate: 0.45
    correlation_with_is: 0.6  # Correlation with in-sample
    
# Execution configuration
coordinator:
  execution_mode: "TRADITIONAL"  # Simple validation
```

**Validation Types**:
- `walk_forward`: Walk-forward analysis with rolling windows
- `out_of_sample`: Simple out-of-sample testing
- `cross_validation`: Time-series cross-validation
- `monte_carlo`: Monte Carlo simulation

**Output Data**:
```python
@dataclass
class ValidationResult:
    validation_metrics: Dict[str, float]
    walk_forward_results: List[Dict[str, Any]]
    success_criteria_met: Dict[str, bool]
    degradation_analysis: Dict[str, float]
    recommendation: str
    
    # Example validation metrics
    validation_metrics = {
        "oos_sharpe_ratio": 0.89,
        "oos_max_drawdown": 0.12,
        "oos_win_rate": 0.52,
        "is_oos_correlation": 0.67,
        "performance_degradation": 0.23  # 23% degradation from IS
    }
    
    # Recommendation
    recommendation = "PROCEED_WITH_CAUTION"  # or "APPROVED", "REJECTED"
```

**Use Cases**:
- Strategy validation
- Out-of-sample testing
- Performance stability assessment
- Production readiness evaluation

### Signal Generation Block

**Block Type**: `signal_generation`

**Description**: Generates and captures trading signals for later analysis or replay

**YAML Configuration**:
```yaml
type: "signal_generation"
name: "Signal Capture"
description: "Generate signals for replay optimization"

# Data configuration
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY", "QQQ"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"

# Strategy configuration
strategies:
  - type: "momentum"
    parameters:
      fast_period: 10
      slow_period: 20
  - type: "mean_reversion"
    parameters:
      rsi_period: 14

# Signal capture configuration
signal_capture:
  capture_all_signals: true
  include_metadata: true
  compression: true
  
  # Output configuration
  output_path: "signals/captured_signals.parquet"
  
  # Metadata to capture
  metadata_fields:
    - "regime_context"
    - "indicator_values"
    - "market_conditions"
    - "volatility_state"

# Execution configuration
coordinator:
  execution_mode: "COMPOSABLE"
  container_pattern: "signal_generation"
```

**Output Data**:
```python
@dataclass
class SignalGenerationResult:
    signals_captured: int
    output_file_path: str
    signal_statistics: Dict[str, Any]
    compression_ratio: float
    metadata: Dict[str, Any]
    
    # Example statistics
    signal_statistics = {
        "total_signals": 8742,
        "signals_per_strategy": {"momentum": 4231, "mean_reversion": 4511},
        "avg_signal_strength": 0.342,
        "avg_confidence": 0.567,
        "file_size_mb": 12.4
    }
```

**Use Cases**:
- Preparing data for signal replay optimization
- Signal database creation
- Multi-strategy signal analysis
- Ensemble method development

### Signal Replay Block

**Block Type**: `signal_replay`

**Description**: Fast optimization using pre-captured signals (10-100x speedup)

**YAML Configuration**:
```yaml
type: "signal_replay"
name: "Fast Parameter Optimization"
description: "Optimize risk parameters using signal replay"

# Input configuration
signal_input:
  source_file: "signals/captured_signals.parquet"
  signal_types: ["momentum", "mean_reversion"]
  
# Optimization configuration (focuses on risk/execution parameters)
optimization:
  method: "bayesian"
  objective: "sharpe_ratio"
  n_trials: 1000  # Much larger due to speed
  
  # Focus on risk management parameters
  parameters:
    position_size_pct: [0.01, 0.05]
    stop_loss_pct: [0.01, 0.05]
    take_profit_pct: [0.02, 0.1]
    max_positions: [3, 10]
    
# Portfolio configuration
portfolio:
  initial_capital: 100000
  commission:
    type: "fixed"
    value: 1.0

# Execution configuration
coordinator:
  execution_mode: "COMPOSABLE"
  container_pattern: "signal_replay"
  parallel_workers: 16  # Higher parallelization
```

**Output Data**:
```python
@dataclass
class SignalReplayResult:
    best_risk_parameters: Dict[str, Any]
    optimization_speed_factor: float  # e.g., 47.3x faster
    trials_completed: int
    best_performance: float
    parameter_sensitivity: Dict[str, float]
    
    # Example results
    best_risk_parameters = {
        "position_size_pct": 0.023,
        "stop_loss_pct": 0.018,
        "take_profit_pct": 0.041,
        "max_positions": 6
    }
    optimization_speed_factor = 47.3  # 47x faster than full backtest
```

**Use Cases**:
- Large-scale parameter optimization
- Risk management tuning
- Execution parameter optimization
- High-frequency strategy development

## 🔄 Block Composition and Data Flow

### Sequential Workflow

**Pattern**: Blocks execute in sequence with data passing between them

```yaml
workflow:
  type: "multi_phase"
  name: "Complete Strategy Development Workflow"
  
phases:
  # Phase 1: Generate signals
  - name: "signal_generation"
    type: "signal_generation"
    config:
      # Signal generation configuration
      data:
        symbols: ["SPY", "QQQ"]
        start_date: "2023-01-01"
        end_date: "2023-10-31"
      strategies:
        - type: "momentum"
        - type: "mean_reversion"
        
  # Phase 2: Fast optimization using signals
  - name: "coarse_optimization"
    type: "signal_replay"
    inputs: ["signal_generation.output_file_path"]
    config:
      optimization:
        method: "grid"
        n_trials: 1000
        
  # Phase 3: Detailed optimization on best candidates
  - name: "fine_optimization"
    type: "optimization"
    inputs: ["coarse_optimization.top_10_parameters"]
    config:
      optimization:
        method: "bayesian"
        n_trials: 100
        
  # Phase 4: Out-of-sample validation
  - name: "validation"
    type: "validation"
    inputs: ["fine_optimization.best_parameters"]
    config:
      data:
        start_date: "2023-11-01"
        end_date: "2023-12-31"
      validation:
        type: "walk_forward"
```

### Parallel Workflow

**Pattern**: Multiple blocks execute in parallel with results aggregated

```yaml
workflow:
  type: "multi_phase"
  name: "Multi-Symbol Strategy Research"
  
phases:
  # Parallel backtests on different symbols
  - name: "parallel_backtests"
    type: "parallel"
    
    blocks:
      - name: "spy_backtest"
        type: "backtest"
        config:
          data:
            symbols: ["SPY"]
          strategies:
            - type: "momentum"
              parameters:
                fast_period: 10
                slow_period: 20
                
      - name: "qqq_backtest"
        type: "backtest"
        config:
          data:
            symbols: ["QQQ"]
          strategies:
            - type: "momentum"
              parameters:
                fast_period: 10
                slow_period: 20
                
      - name: "iwm_backtest"
        type: "backtest"
        config:
          data:
            symbols: ["IWM"]
          strategies:
            - type: "momentum"
              parameters:
                fast_period: 10
                slow_period: 20
                
  # Aggregate results
  - name: "results_aggregation"
    type: "analysis"
    inputs: ["parallel_backtests.*"]
    config:
      analysis:
        type: "performance_comparison"
        aggregation_method: "weighted_average"
```

### Conditional Workflow

**Pattern**: Blocks execute based on conditions from previous blocks

```yaml
workflow:
  type: "multi_phase"
  name: "Adaptive Strategy Development"
  
phases:
  # Initial backtest
  - name: "initial_backtest"
    type: "backtest"
    config:
      # Backtest configuration
      
  # Conditional optimization based on initial performance
  - name: "optimization"
    type: "optimization"
    condition: "initial_backtest.sharpe_ratio > 0.5"  # Only optimize if promising
    inputs: ["initial_backtest.strategy_config"]
    config:
      optimization:
        method: "bayesian"
        n_trials: 200
        
  # Alternative analysis if optimization skipped
  - name: "alternative_analysis"
    type: "analysis"
    condition: "initial_backtest.sharpe_ratio <= 0.5"  # If not promising
    inputs: ["initial_backtest.signals"]
    config:
      analysis:
        type: "signal_analysis"
        
  # Final validation (conditional on optimization success)
  - name: "validation"
    type: "validation"
    condition: "optimization.best_performance > 1.0"
    inputs: ["optimization.best_parameters"]
    config:
      validation:
        type: "walk_forward"
```

## 📊 Block Performance Characteristics

### Execution Speed Comparison

```
Block Type Performance (relative to basic backtest):

backtest:           1.0x   (baseline)
optimization:       10-50x  (depending on trials)
signal_generation:  1.2x   (slight overhead for capture)
signal_replay:      0.01-0.1x  (10-100x faster)
analysis:           0.8x   (no execution, just analysis)
validation:         2-5x   (multiple test periods)
```

### Memory Usage

```
Block Type Memory Usage:

backtest:           100-300MB
optimization:       200-500MB per trial (parallel)
signal_generation:  150-400MB + signal storage
signal_replay:      50-150MB (minimal data)
analysis:           100-200MB
validation:         150-350MB
```

### Recommended Use Cases by Scale

| Scale | Primary Blocks | Pattern | Performance Focus |
|-------|---------------|---------|-------------------|
| Small (< 100 trials) | `backtest`, `analysis` | Sequential | Simplicity |
| Medium (100-1000 trials) | `optimization`, `validation` | Sequential | Accuracy |
| Large (1000+ trials) | `signal_generation` → `signal_replay` | Hybrid | Speed |
| Research | `analysis`, `signal_generation` | Parallel | Flexibility |

## 🔧 Custom Block Development

### Creating Custom Blocks

```python
from src.core.workflows import WorkflowBlock
from typing import Dict, Any

class CustomAnalysisBlock(WorkflowBlock):
    """Custom analysis block implementation."""
    
    block_type = "custom_analysis"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.analysis_type = config.get("analysis_type", "default")
        
    async def execute(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute custom analysis."""
        
        # Load data
        data = await self.load_data()
        
        # Perform custom analysis
        if self.analysis_type == "correlation_analysis":
            results = await self.correlation_analysis(data)
        elif self.analysis_type == "regime_detection":
            results = await self.regime_detection(data)
        else:
            results = await self.default_analysis(data)
            
        # Return results
        return {
            "analysis_results": results,
            "metadata": self.get_metadata(),
            "output_files": self.get_output_files()
        }
        
    async def correlation_analysis(self, data) -> Dict[str, Any]:
        """Custom correlation analysis implementation."""
        # Implementation here
        pass
        
    def validate_config(self) -> bool:
        """Validate block configuration."""
        required_fields = ["data", "analysis_type"]
        return all(field in self.config for field in required_fields)
```

### Registering Custom Blocks

```python
from src.core.workflows import WorkflowRegistry

# Register custom block
registry = WorkflowRegistry()
registry.register_block_type(
    block_type="custom_analysis",
    block_class=CustomAnalysisBlock,
    config_schema={
        "type": "object",
        "properties": {
            "analysis_type": {"type": "string"},
            "data": {"type": "object"},
            "output_config": {"type": "object"}
        },
        "required": ["analysis_type", "data"]
    }
)
```

## 🤔 Common Questions

**Q: Can blocks be reused in different workflows?**
A: Yes, blocks are designed to be composable and reusable across different workflow configurations.

**Q: How do I pass data between blocks?**
A: Use the `inputs` configuration to reference outputs from previous blocks. Data is automatically serialized and passed.

**Q: Can blocks run in parallel?**
A: Yes, blocks without dependencies can run in parallel. Use the `parallel` workflow type.

**Q: What happens if a block fails?**
A: Depends on error handling configuration. Options include fail-fast, continue with warnings, or retry.

**Q: How do I optimize block performance?**
A: Use appropriate container patterns, enable parallelization, and consider signal replay for optimization-heavy workflows.

---

Continue to [Performance Benchmarks](performance-benchmarks.md) for system performance specifications →