# Configuration Schema

Complete YAML configuration reference for ADMF-PC. This document provides the exact schema, validation rules, and examples for all configuration options.

## 📋 Root Configuration Schema

### Top-Level Structure

```yaml
# Required sections
workflow:                    # Workflow configuration (required)
  type: string              # Workflow type (required)
  name: string              # Workflow name (optional)
  description: string       # Workflow description (optional)

data:                       # Data configuration (required)
  symbols: list[string]     # List of symbols to trade (required)
  start_date: string        # Start date (required for historical)
  end_date: string          # End date (optional, defaults to latest)

# Core strategy configuration
strategies:                 # List of strategies (required)
  - type: string           # Strategy type (required)
    params: dict           # Strategy parameters (optional)
    allocation: float      # Portfolio allocation (optional, default: equal weight)

# Optional sections with defaults
risk_management:           # Risk management configuration (optional)
  type: string            # Risk manager type (optional, default: "basic")
  params: dict            # Risk parameters (optional)

execution:                 # Execution configuration (optional)
  type: string            # Execution type (optional, default: "market")
  params: dict            # Execution parameters (optional)

portfolio:                 # Portfolio configuration (optional)
  initial_capital: float  # Starting capital (optional, default: 100000)
  currency: string        # Base currency (optional, default: "USD")

reporting:                 # Reporting configuration (optional)
  output_path: string     # Report output path (optional)
  include_charts: bool    # Include charts in reports (optional, default: true)

# Advanced sections
optimization:              # Optimization configuration (conditional)
infrastructure:            # Infrastructure configuration (optional)
monitoring:               # Monitoring configuration (optional)
```

## 🎼 Workflow Configuration

### Workflow Types and Schema

#### Backtest Workflow
```yaml
workflow:
  type: "backtest"                    # Required
  name: string                        # Optional
  description: string                 # Optional
  
  # Backtest-specific options
  execution_mode: enum                # Optional: "TRADITIONAL", "AUTO", "COMPOSABLE", "HYBRID"
  signal_capture: bool                # Optional: Capture signals for replay (default: false)
  signal_output_path: string          # Conditional: Required if signal_capture=true
  
  # Performance options
  parallel_execution: bool            # Optional: Enable parallel execution (default: true)
  container_count: int                # Optional: Number of containers (default: auto)
  
validation:
  type: "backtest"
  required_fields: ["type"]
  optional_fields: ["name", "description", "execution_mode", "signal_capture"]
  enum_values:
    execution_mode: ["TRADITIONAL", "AUTO", "COMPOSABLE", "HYBRID"]
```

#### Optimization Workflow
```yaml
workflow:
  type: "optimization"
  name: string                        # Optional
  description: string                 # Optional
  
  # Optimization-specific options
  signal_replay: bool                 # Optional: Use signal replay for speed (default: false)
  signal_input_path: string           # Conditional: Required if signal_replay=true
  
optimization:                         # Required for optimization workflows
  method: enum                        # Required: "grid", "random", "bayesian", "genetic"
  objective: string                   # Required: Optimization objective
  parameters: dict                    # Required: Parameters to optimize
  
  # Method-specific options
  n_trials: int                       # Conditional: Required for random/bayesian
  max_workers: int                    # Optional: Parallel workers (default: auto)
  timeout_hours: float                # Optional: Maximum runtime (default: no limit)
  
validation:
  type: "optimization"
  required_fields: ["type", "optimization"]
  enum_values:
    method: ["grid", "random", "bayesian", "genetic", "differential_evolution"]
  conditional_requirements:
    - if: "method in ['random', 'bayesian']"
      then: "n_trials required"
```

#### Multi-Phase Workflow
```yaml
workflow:
  type: "multi_phase"
  name: string                        # Optional
  description: string                 # Optional
  
phases:                              # Required for multi_phase workflows
  - name: string                     # Required: Phase name
    type: enum                       # Required: "optimization", "backtest", "analysis", "validation"
    config: dict                     # Required: Phase-specific configuration
    inputs: list[string]             # Optional: References to previous phase outputs
    outputs: list[string]            # Optional: Outputs this phase produces
    
    # Resource management
    container_count: int             # Optional: Containers for this phase
    timeout_hours: float             # Optional: Phase timeout
    
validation:
  type: "multi_phase"
  required_fields: ["type", "phases"]
  phase_validation:
    required_fields: ["name", "type", "config"]
    enum_values:
      type: ["optimization", "backtest", "analysis", "validation", "signal_generation", "signal_replay"]
```

#### Walk-Forward Workflow
```yaml
workflow:
  type: "walk_forward"
  name: string                        # Optional
  description: string                 # Optional
  
walk_forward:                        # Required for walk_forward workflows
  train_period_days: int             # Required: Training window size (30-2000)
  test_period_days: int              # Required: Testing window size (5-500)
  step_days: int                     # Required: Step size (1-200)
  
  # Window configuration
  window_type: enum                  # Optional: "fixed", "expanding" (default: "fixed")
  min_train_days: int                # Conditional: Required if window_type="expanding"
  
  # Optimization configuration
  optimization: dict                 # Required: Optimization settings per window
  
validation:
  type: "walk_forward"
  required_fields: ["type", "walk_forward"]
  value_ranges:
    train_period_days: [30, 2000]
    test_period_days: [5, 500]
    step_days: [1, 200]
  enum_values:
    window_type: ["fixed", "expanding", "percentage", "adaptive"]
```

## 📊 Data Configuration

### Data Source Schema

#### CSV Data Source
```yaml
data:
  source:
    type: "csv"                      # Required
    path: string                     # Required: Path to CSV file
    
    # Column mapping
    symbol_column: string            # Optional: Symbol column name (default: "symbol")
    timestamp_column: string         # Optional: Timestamp column name (default: "timestamp")
    price_columns: dict              # Optional: OHLCV column mapping
    
    # Data processing
    date_format: string              # Optional: Timestamp format (default: "%Y-%m-%d %H:%M:%S")
    timezone: string                 # Optional: Timezone (default: "UTC")
    
  # Time range
  symbols: list[string]              # Required: List of symbols
  start_date: string                 # Required: Start date (YYYY-MM-DD)
  end_date: string                   # Optional: End date (default: latest)
  
  # Data filtering
  timeframe: string                  # Optional: Bar timeframe (default: "1m")
  max_bars: int                      # Optional: Limit number of bars
  
validation:
  required_fields: ["source.type", "source.path", "symbols", "start_date"]
  date_format: "YYYY-MM-DD"
  supported_timeframes: ["1s", "1m", "5m", "15m", "30m", "1h", "4h", "1d"]
```

#### Live Data Source
```yaml
data:
  source:
    type: "live"                     # Required
    provider: enum                   # Required: Data provider
    
    # Authentication
    api_credentials: dict            # Required: Provider-specific credentials
    environment: enum                # Optional: "paper", "live" (default: "paper")
    
    # Connection settings
    connection_timeout: int          # Optional: Connection timeout seconds (default: 30)
    reconnect_attempts: int          # Optional: Reconnection attempts (default: 3)
    buffer_size: int                 # Optional: Internal buffer size (default: 1000)
    
  # Trading configuration
  symbols: list[string]              # Required: List of symbols
  timeframe: string                  # Optional: Bar timeframe (default: "1m")
  
  # Market hours
  market_hours:                      # Optional: Trading hours configuration
    start_time: string              # Optional: Market start time (default: "09:30")
    end_time: string                # Optional: Market end time (default: "16:00")
    timezone: string                # Optional: Market timezone (default: "US/Eastern")
    
validation:
  required_fields: ["source.type", "source.provider", "source.api_credentials", "symbols"]
  enum_values:
    provider: ["alpaca", "interactive_brokers", "polygon", "td_ameritrade"]
    environment: ["paper", "live"]
  time_format: "HH:MM"
```

#### Database Data Source
```yaml
data:
  source:
    type: "database"                 # Required
    connection_string: string        # Required: Database connection string
    table: string                    # Required: Table name
    
    # Column configuration
    symbol_column: string            # Optional: Symbol column (default: "symbol")
    timestamp_column: string         # Optional: Timestamp column (default: "timestamp")
    price_columns: dict              # Optional: OHLCV column mapping
    
    # Query configuration
    query_template: string           # Optional: Custom query template
    batch_size: int                  # Optional: Query batch size (default: 10000)
    
  symbols: list[string]              # Required: List of symbols
  start_date: string                 # Required: Start date
  end_date: string                   # Optional: End date
  
validation:
  required_fields: ["source.type", "source.connection_string", "source.table", "symbols", "start_date"]
  connection_string_format: "dialect://user:password@host:port/database"
```

## 🎯 Strategy Configuration

### Strategy Schema

```yaml
strategies:
  - type: string                     # Required: Strategy type from component catalog
    name: string                     # Optional: Custom strategy name
    enabled: bool                    # Optional: Enable/disable strategy (default: true)
    allocation: float                # Optional: Portfolio allocation (0.0-1.0, default: equal)
    
    # Strategy parameters
    params: dict                     # Optional: Strategy-specific parameters
    
    # Data configuration
    symbols: list[string]            # Optional: Strategy-specific symbols (default: use global)
    timeframe: string                # Optional: Strategy-specific timeframe
    
    # Risk overrides
    risk_overrides:                  # Optional: Strategy-specific risk settings
      max_position_size: float       # Optional: Override max position size
      stop_loss_pct: float           # Optional: Override stop loss
      
    # Optimization configuration
    optimization_target: bool        # Optional: Include in optimization (default: false)
    
validation:
  required_fields: ["type"]
  value_ranges:
    allocation: [0.0, 1.0]
  sum_constraint:
    field: "allocation"
    max_sum: 1.0
  type_validation:
    type: "must exist in component catalog"
```

### Multi-Strategy Configuration

```yaml
strategies:
  # Multiple strategies with different allocations
  - type: "momentum"
    name: "Fast_Momentum"
    allocation: 0.4                  # 40% allocation
    params:
      fast_period: 5
      slow_period: 15
      
  - type: "mean_reversion"
    name: "RSI_MeanRev" 
    allocation: 0.3                  # 30% allocation
    params:
      period: 14
      oversold: 30
      overbought: 70
      
  - type: "sklearn_model"
    name: "ML_Strategy"
    allocation: 0.3                  # 30% allocation
    params:
      model_path: "models/rf.pkl"
      feature_columns: ["rsi", "macd"]
      
validation:
  allocation_sum: 1.0                # Total allocations must sum to 1.0
  max_strategies: 20                 # Maximum number of strategies
```

## 🛡️ Risk Management Schema

### Basic Risk Management

```yaml
risk_management:
  type: string                       # Required: Risk manager type
  
  # Common parameters
  params:
    position_size_pct: float         # Required: Position size as % of portfolio (0.001-1.0)
    max_positions: int               # Optional: Max concurrent positions (default: 10)
    max_exposure_pct: float          # Optional: Max gross exposure (default: 1.0)
    
    # Stop loss/take profit
    stop_loss_pct: float             # Optional: Stop loss percentage (0.001-0.5)
    take_profit_pct: float           # Optional: Take profit percentage (0.001-2.0)
    
    # Portfolio limits
    max_drawdown_pct: float          # Optional: Max drawdown before action (0.01-0.5)
    daily_loss_limit_pct: float      # Optional: Daily loss limit (0.001-0.2)
    
validation:
  required_fields: ["type", "params.position_size_pct"]
  value_ranges:
    position_size_pct: [0.001, 1.0]
    max_exposure_pct: [0.1, 5.0]
    stop_loss_pct: [0.001, 0.5]
    take_profit_pct: [0.001, 2.0]
    max_drawdown_pct: [0.01, 0.5]
```

### Advanced Risk Management

```yaml
risk_management:
  type: "advanced"
  
  # Multi-level risk controls
  position_limits:                   # Position-level controls
    max_position_size_usd: float     # Absolute dollar limit
    position_size_pct: float         # Percentage limit
    max_leverage: float              # Maximum leverage
    
  portfolio_limits:                  # Portfolio-level controls
    max_gross_exposure: float        # Maximum gross exposure
    max_net_exposure: float          # Maximum net exposure
    max_correlation: float           # Maximum position correlation
    sector_limits: dict              # Per-sector limits
    
  time_controls:                     # Time-based controls
    max_holding_period_days: int     # Maximum holding period
    no_overnight_positions: bool     # Close before market close
    trading_hours_only: bool         # Trade only during market hours
    
  volatility_controls:               # Volatility-based controls
    max_symbol_volatility: float     # Maximum symbol volatility
    portfolio_volatility_limit: float # Portfolio volatility limit
    volatility_scaling: bool         # Scale size with volatility
    
validation:
  advanced_validation:
    sector_limits_sum: 1.0           # Sector limits cannot exceed 100%
    leverage_consistency: true       # Ensure leverage limits are consistent
```

## ⚙️ Execution Configuration

### Execution Schema

```yaml
execution:
  type: string                       # Optional: Execution type (default: "market")
  
  # Order configuration
  order_type: enum                   # Optional: Default order type (default: "market")
  execution_delay_seconds: int       # Optional: Delay before execution (default: 0)
  
  # Cost modeling
  slippage_bps: float               # Optional: Slippage in basis points (default: 10)
  commission_per_share: float       # Optional: Commission per share (default: 0.01)
  commission_per_trade: float       # Optional: Fixed commission per trade
  
  # Market impact
  market_impact:                    # Optional: Market impact modeling
    enabled: bool                   # Enable market impact (default: false)
    model: enum                     # Model type: "linear", "sqrt", "almgren_chriss"
    participation_rate: float       # Participation rate (default: 0.1)
    
  # Advanced execution
  smart_routing:                    # Optional: Smart order routing
    enabled: bool                   # Enable smart routing (default: false)
    venues: list[string]            # List of execution venues
    routing_algorithm: enum         # Routing algorithm: "price", "liquidity", "speed"
    
validation:
  enum_values:
    order_type: ["market", "limit", "stop", "stop_limit"]
    market_impact.model: ["linear", "sqrt", "almgren_chriss"]
    smart_routing.routing_algorithm: ["price", "liquidity", "speed"]
  value_ranges:
    slippage_bps: [0, 500]
    commission_per_share: [0, 1.0]
    market_impact.participation_rate: [0.01, 0.5]
```

## 💰 Portfolio Configuration

### Portfolio Schema

```yaml
portfolio:
  initial_capital: float             # Required: Starting capital (>0)
  currency: string                   # Optional: Base currency (default: "USD")
  
  # Cash management
  cash_management:                   # Optional: Cash management settings
    min_cash_pct: float             # Minimum cash percentage (default: 0.05)
    max_cash_pct: float             # Maximum cash percentage (default: 0.5)
    reinvest_dividends: bool        # Reinvest dividends (default: true)
    interest_rate: float            # Interest on cash (default: 0.02)
    
  # Multi-currency support
  currency_hedging:                  # Optional: Currency hedging
    enabled: bool                   # Enable currency hedging (default: false)
    hedge_ratio: float              # Hedge ratio (default: 0.8)
    rebalance_frequency: enum       # Rebalancing frequency
    
  # Leverage and margin
  leverage:                          # Optional: Leverage configuration
    enabled: bool                   # Enable leverage (default: false)
    max_leverage: float             # Maximum leverage ratio (default: 1.0)
    margin_rate: float              # Annual margin rate (default: 0.05)
    maintenance_margin: float       # Maintenance margin requirement (default: 0.25)
    
validation:
  required_fields: ["initial_capital"]
  value_ranges:
    initial_capital: [1, 1e12]
    cash_management.min_cash_pct: [0.0, 0.5]
    cash_management.max_cash_pct: [0.1, 1.0]
    leverage.max_leverage: [1.0, 10.0]
    leverage.margin_rate: [0.0, 0.2]
  currency_codes: ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"]
```

## 📈 Optimization Configuration

### Optimization Schema

```yaml
optimization:
  method: enum                       # Required: Optimization method
  objective: string                  # Required: Optimization objective
  parameters: dict                   # Required: Parameters to optimize
  
  # Resource configuration
  max_workers: int                   # Optional: Maximum parallel workers (default: auto)
  timeout_hours: float               # Optional: Maximum runtime hours (default: no limit)
  
  # Method-specific configuration
  grid_options:                      # Conditional: For grid search
    exhaustive: bool                 # Test all combinations (default: true)
    
  random_options:                    # Conditional: For random search
    n_trials: int                    # Required: Number of trials
    seed: int                        # Optional: Random seed for reproducibility
    
  bayesian_options:                  # Conditional: For Bayesian optimization
    n_trials: int                    # Required: Number of trials
    n_initial_points: int            # Optional: Initial random points (default: 10)
    acquisition_function: enum       # Optional: Acquisition function (default: "ei")
    
  # Constraints
  constraints:                       # Optional: Optimization constraints
    - type: enum                     # Constraint type: "parameter", "performance", "risk"
      constraint: string             # Constraint expression
      
validation:
  required_fields: ["method", "objective", "parameters"]
  enum_values:
    method: ["grid", "random", "bayesian", "genetic", "differential_evolution"]
    acquisition_function: ["ei", "pi", "ucb"]
  conditional_requirements:
    - if: "method in ['random', 'bayesian']"
      then: "n_trials required"
```

### Parameter Specification

```yaml
optimization:
  parameters:
    # Discrete parameters (for grid search)
    fast_period: [5, 10, 15, 20]     # List of discrete values
    slow_period: [20, 30, 40, 50]
    
    # Continuous parameters (for continuous optimization)
    signal_threshold: [0.001, 0.1]   # [min, max] range
    position_size: [0.01, 0.1]
    
    # Categorical parameters
    ma_type: ["sma", "ema", "wma"]    # List of categorical options
    
validation:
  parameter_types:
    discrete: "list of values"
    continuous: "[min, max] range"
    categorical: "list of options"
  parameter_ranges:
    all_numeric_parameters: "> 0"
    percentage_parameters: "[0.0, 1.0]"
```

## 🏗️ Infrastructure Configuration

### Infrastructure Schema

```yaml
infrastructure:
  # Resource limits
  max_workers: int                   # Optional: Maximum worker processes (default: auto)
  max_memory_gb: float               # Optional: Maximum memory usage (default: auto)
  max_cpu_cores: int                 # Optional: Maximum CPU cores (default: auto)
  
  # Container configuration
  container_pool_size: int           # Optional: Container pool size (default: 100)
  container_timeout_seconds: int     # Optional: Container timeout (default: 3600)
  
  # Performance tuning
  performance:                       # Optional: Performance tuning
    enable_jit: bool                 # Enable JIT compilation (default: true)
    parallel_indicators: bool        # Parallel indicator calculation (default: true)
    memory_mapping: bool             # Use memory mapping for large datasets (default: false)
    
  # Monitoring
  monitoring:                        # Optional: Monitoring configuration
    enabled: bool                    # Enable monitoring (default: true)
    metrics_interval_seconds: int    # Metrics collection interval (default: 10)
    log_level: enum                  # Logging level (default: "INFO")
    
validation:
  value_ranges:
    max_workers: [1, 1000]
    max_memory_gb: [1, 1000]
    max_cpu_cores: [1, 128]
    container_pool_size: [1, 10000]
  enum_values:
    log_level: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
```

## 📊 Reporting Configuration

### Reporting Schema

```yaml
reporting:
  # Output configuration
  output_path: string                # Optional: Report output directory
  output_formats: list[enum]         # Optional: Output formats (default: ["html"])
  
  # Report content
  include_charts: bool               # Optional: Include charts (default: true)
  include_trade_log: bool            # Optional: Include detailed trade log (default: false)
  include_daily_metrics: bool        # Optional: Include daily metrics (default: false)
  
  # Benchmarking
  benchmarks:                        # Optional: Benchmark comparisons
    - symbol: string                 # Benchmark symbol
      name: string                   # Benchmark name
      
  # Custom reporting
  custom_metrics:                    # Optional: Custom performance metrics
    - name: string                   # Metric name
      function: string               # Function to calculate metric
      
validation:
  enum_values:
    output_formats: ["html", "pdf", "json", "csv"]
  path_validation:
    output_path: "must be valid directory path"
```

## 🔧 Complete Configuration Example

### Comprehensive Configuration

```yaml
# Complete ADMF-PC configuration example
workflow:
  type: "multi_phase"
  name: "Comprehensive Strategy Research"
  description: "Multi-phase optimization and validation workflow"
  
data:
  source:
    type: "csv"
    path: "data/SPY_1m.csv"
    timezone: "US/Eastern"
  symbols: ["SPY", "QQQ", "IWM"]
  start_date: "2022-01-01"
  end_date: "2023-12-31"
  timeframe: "1h"

phases:
  - name: "parameter_optimization"
    type: "optimization"
    config:
      strategies:
        - type: "momentum"
          optimization_target: true
      optimization:
        method: "bayesian"
        n_trials: 200
        objective: "sharpe_ratio"
        parameters:
          fast_period: [5, 30]
          slow_period: [20, 100]
          signal_threshold: [0.001, 0.1]
    container_count: 50
    
  - name: "validation"
    type: "walk_forward"
    inputs: ["parameter_optimization.best_parameters"]
    config:
      walk_forward:
        train_period_days: 252
        test_period_days: 63
        step_days: 21

strategies:
  - type: "momentum"
    allocation: 0.6
  - type: "mean_reversion"
    allocation: 0.4

risk_management:
  type: "volatility_based"
  params:
    target_volatility: 0.15
    max_position_size: 0.05
    max_drawdown_pct: 0.12

execution:
  slippage_bps: 10
  commission_per_share: 0.005
  market_impact:
    enabled: true
    model: "linear"
    participation_rate: 0.1

portfolio:
  initial_capital: 100000
  currency: "USD"
  cash_management:
    min_cash_pct: 0.05
    reinvest_dividends: true

infrastructure:
  max_workers: 16
  max_memory_gb: 32
  monitoring:
    enabled: true
    log_level: "INFO"

reporting:
  output_path: "reports/comprehensive_research/"
  output_formats: ["html", "json"]
  include_charts: true
  include_trade_log: true
  benchmarks:
    - symbol: "SPY"
      name: "S&P 500"
```

## ✅ Configuration Validation

### Validation Rules

**Required Field Validation**:
- All required fields must be present
- Required fields cannot be null or empty

**Type Validation**:
- All fields must match specified types
- Numeric fields validated for proper number format
- Date fields validated for proper date format

**Range Validation**:
- Numeric fields validated against min/max ranges
- Percentage fields validated to be between 0.0 and 1.0
- Enum fields validated against allowed values

**Cross-Field Validation**:
- Sum constraints (e.g., strategy allocations sum to 1.0)
- Conditional requirements (e.g., n_trials required for random optimization)
- Logical consistency (e.g., start_date before end_date)

**Business Logic Validation**:
- Component types must exist in component catalog
- Parameter combinations must be logically valid
- Resource requirements must be achievable

### Common Validation Errors

```yaml
# Common configuration errors and fixes

# Error: Invalid enum value
workflow:
  type: "invalid_type"              # ❌ Not a valid workflow type
# Fix:
workflow:
  type: "backtest"                  # ✅ Valid workflow type

# Error: Value out of range  
risk_management:
  params:
    position_size_pct: 2.0          # ❌ 200% position size invalid
# Fix:
risk_management:
  params:
    position_size_pct: 0.02         # ✅ 2% position size

# Error: Missing required field
optimization:
  method: "bayesian"                # ❌ Missing required n_trials
# Fix:
optimization:
  method: "bayesian"
  n_trials: 100                     # ✅ Required field added

# Error: Invalid date format
data:
  start_date: "01/01/2023"          # ❌ Wrong date format
# Fix:
data:
  start_date: "2023-01-01"          # ✅ Correct YYYY-MM-DD format
```

## 🔍 Schema Validation Tools

### Validate Configuration

```bash
# Validate configuration file
python -c "
from src.core.config import validate_config
result = validate_config('config/my_config.yaml')
if result.is_valid:
    print('Configuration is valid!')
else:
    print('Validation errors:')
    for error in result.errors:
        print(f'  - {error}')
"

# Validate specific section
python -c "
from src.core.config import validate_section
validate_section('strategies', my_strategies_config)
"
```

---

Continue to [Event Reference](event-reference.md) for complete event system specifications →