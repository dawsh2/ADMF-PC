# ADMF-PC YAML Configuration Guide

## Getting Started in 5 Minutes

ADMF-PC is a **zero-code** trading system. Everything is configured through YAML files. No programming required.

### Quick Start Example

1. **Create a simple backtest configuration** (`my_first_backtest.yaml`):

```yaml
workflow:
  type: "backtest"
  name: "My First Strategy Test"

data:
  symbols: ["AAPL", "GOOGL"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"

strategies:
  - name: "simple_momentum"
    type: "momentum"
    fast_period: 10
    slow_period: 30

risk:
  initial_capital: 100000
  position_size_pct: 2.0
  max_drawdown_pct: 10.0
```

2. **Run it**:
```bash
admf-run my_first_backtest.yaml
```

3. **Get results** in `results/my_first_backtest/`

That's it! No code, no complex setup. Just describe what you want to test in YAML.

---

## Table of Contents

1. [Basic Concepts](#basic-concepts)
2. [Workflow Types](#workflow-types)  
3. [Data Configuration](#data-configuration)
4. [Strategy Configuration](#strategy-configuration)
5. [Risk & Portfolio Management](#risk--portfolio-management)
6. [Infrastructure Setup](#infrastructure-setup)
7. [Multi-Phase Optimization](#multi-phase-optimization)
8. [Live Trading Configuration](#live-trading-configuration)
9. [Complete Examples](#complete-examples)
10. [Troubleshooting](#troubleshooting)

---

## Basic Concepts

### The YAML Structure

Every ADMF-PC configuration follows this basic structure:

```yaml
# What type of analysis to run
workflow:
  type: "backtest" | "optimization" | "live_trading" | "analysis"
  name: "Human readable name"
  description: "Optional description"

# Where to get data
data:
  symbols: [list of symbols]
  timeframe: "1D" | "1H" | "5M" | etc.
  start_date: "YYYY-MM-DD"
  end_date: "YYYY-MM-DD"

# What strategies to test  
strategies:
  - name: "strategy_1"
    type: "momentum" | "mean_reversion" | "breakout" | etc.
    # strategy-specific parameters

# How to manage risk
risk:
  initial_capital: 100000
  position_sizing: { ... }
  limits: [ ... ]

# Where to save results
output:
  path: "results/my_test/"
  format: ["json", "csv", "html"]
```

### Key Principles

1. **Declarative**: Describe *what* you want, not *how* to do it
2. **Hierarchical**: Complex strategies built from simple components  
3. **Reusable**: Save and share configurations
4. **Environment-aware**: Use variables for different environments

---

## Workflow Types

### 1. Backtest (`type: "backtest"`)

Test strategies on historical data.

```yaml
workflow:
  type: "backtest"
  name: "Strategy Performance Test"

# Basic backtest - just add data and strategies
```

### 2. Optimization (`type: "optimization"`)  

Find the best parameters for your strategies.

```yaml
workflow:
  type: "optimization"
  name: "Parameter Optimization"

optimization:
  algorithm: "grid_search"  # or "genetic", "bayesian"
  objective: "sharpe_ratio"  # what to maximize
  
  # Define parameter ranges to test
  parameter_space:
    fast_period: [5, 10, 15, 20]
    slow_period: [20, 30, 40, 50]
```

### 3. Live Trading (`type: "live_trading"`)

Deploy strategies to live markets.

```yaml
workflow:
  type: "live_trading" 
  name: "Production Trading"

broker:
  name: "interactive_brokers"
  account: "${ACCOUNT_ID}"
  paper_trading: true  # Start with paper trading!
```

### 4. Analysis (`type: "analysis"`)

Research and analyze without trading.

```yaml
workflow:
  type: "analysis"
  name: "Market Research"

analysis:
  type: "signal_quality"  # Analyze signal performance
  # or "regime_analysis", "correlation_study", etc.
```

---

## Data Configuration

### Basic Data Setup

```yaml
data:
  # Required: What symbols to analyze
  symbols: ["AAPL", "GOOGL", "MSFT", "SPY"]
  
  # Required: Time period
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  
  # Required: Data frequency
  timeframe: "1D"  # Daily bars
  # Options: "1M", "5M", "15M", "1H", "4H", "1D", "1W"
```

### Advanced Data Sources

```yaml
data:
  # Multiple data sources
  sources:
    primary:
      type: "csv"
      path: "data/stocks/"
      date_format: "%Y-%m-%d"
    
    fundamental:
      type: "database" 
      connection: "sqlite:///data/fundamentals.db"
      table: "earnings"
    
    alternative:
      type: "api"
      provider: "alpha_vantage"
      api_key: "${ALPHA_VANTAGE_KEY}"
  
  # Data preprocessing
  preprocessing:
    fill_missing: "forward_fill"  # Handle missing data
    outlier_detection: true
    corporate_actions: true  # Adjust for splits/dividends
```

### Data Validation

```yaml
data:
  symbols: ["AAPL", "GOOGL"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  
  # Ensure data quality
  validation:
    min_data_points: 100  # Require minimum history
    max_missing_pct: 5    # Max 5% missing data allowed
    require_volume: true   # Must have volume data
```

---

## Strategy Configuration

### Pre-Built Strategies

ADMF-PC includes ready-to-use strategies. Just specify parameters:

#### Momentum Strategy
```yaml
strategies:
  - name: "momentum_1"
    type: "momentum"
    
    # Strategy parameters
    fast_period: 10      # Fast moving average
    slow_period: 30      # Slow moving average  
    signal_threshold: 0.02  # Minimum signal strength
    
    # Which symbols this strategy trades
    symbols: ["AAPL", "GOOGL"]  # Optional: default is all symbols
```

#### Mean Reversion Strategy
```yaml
strategies:
  - name: "mean_reversion_1"
    type: "mean_reversion"
    
    lookback_period: 20    # Analysis window
    std_threshold: 2.0     # Standard deviation threshold
    hold_period: 5         # Days to hold position
    
    symbols: ["SPY"]       # ETFs work well for mean reversion
```

#### Breakout Strategy  
```yaml
strategies:
  - name: "breakout_1"
    type: "breakout"
    
    breakout_period: 20    # Lookback for high/low
    volume_threshold: 1.5  # Volume confirmation
    atr_multiplier: 2.0    # Stop loss distance
    
    symbols: ["QQQ", "SPY"]
```

#### Pairs Trading
```yaml
strategies:
  - name: "pairs_1"
    type: "pairs_trading"
    
    # Define the pair
    symbol_1: "AAPL"
    symbol_2: "MSFT"
    
    # Pair parameters
    lookback_period: 60
    entry_threshold: 2.0   # Standard deviations
    exit_threshold: 0.5
```

### Multi-Strategy Configurations

```yaml
# Run multiple strategies together
strategies:
  - name: "momentum_tech"
    type: "momentum"
    fast_period: 10
    slow_period: 30
    symbols: ["AAPL", "GOOGL", "MSFT"]
    
  - name: "mean_reversion_etf"
    type: "mean_reversion" 
    lookback_period: 20
    symbols: ["SPY", "QQQ"]
    
  - name: "breakout_crypto"
    type: "breakout"
    breakout_period: 14
    symbols: ["BTC-USD", "ETH-USD"]

# Optional: How to weight strategies
strategy_allocation:
  momentum_tech: 0.5      # 50% of capital
  mean_reversion_etf: 0.3 # 30% of capital  
  breakout_crypto: 0.2    # 20% of capital
```

---

## Risk & Portfolio Management

### Basic Risk Controls

```yaml
risk:
  # Starting capital
  initial_capital: 100000
  
  # Position sizing
  position_size_pct: 2.0  # 2% of capital per position
  max_position_size: 10000  # Never more than $10k per position
  
  # Portfolio limits
  max_portfolio_risk: 20.0  # Max 20% of capital at risk
  max_drawdown_pct: 15.0    # Stop trading at 15% drawdown
  
  # Daily limits
  max_daily_loss: 2000      # Stop trading if lose $2k in a day
  max_trades_per_day: 20    # Limit number of trades
```

### Advanced Risk Management

```yaml
risk:
  initial_capital: 100000
  
  # Multiple position sizing methods
  position_sizing:
    # Default method
    default:
      type: "percentage"
      percentage: 2.0
    
    # High volatility stocks - smaller positions
    high_vol:
      type: "volatility_based"
      risk_per_trade: 1.0  # 1% risk per trade
      symbols: ["TSLA", "NVDA"]
    
    # ETFs - larger positions allowed
    etf:
      type: "fixed_amount"
      amount: 5000
      symbols: ["SPY", "QQQ", "IWM"]
  
  # Multiple risk limits
  limits:
    # Overall portfolio
    - type: "max_exposure"
      limit_pct: 25.0
    
    # Per sector  
    - type: "sector_exposure"
      sector: "technology"
      limit_pct: 40.0
    
    # Correlation limit
    - type: "correlation_limit"
      max_correlation: 0.7
      
    # Concentration limit
    - type: "single_position_limit"
      limit_pct: 5.0  # No position > 5% of portfolio
```

### Risk by Market Regime

```yaml
# Adaptive risk based on market conditions
risk:
  initial_capital: 100000
  
  # Different risk profiles for different market regimes
  regime_based_risk:
    bull_market:
      position_size_pct: 3.0  # More aggressive in bull markets
      max_portfolio_risk: 30.0
      
    bear_market:
      position_size_pct: 1.0  # Conservative in bear markets
      max_portfolio_risk: 10.0
      
    high_volatility:
      position_size_pct: 0.5  # Very conservative in volatile periods
      max_portfolio_risk: 5.0
```

---

## Infrastructure Setup

### Technical Indicators

```yaml
# Define indicators used by strategies
indicators:
  # Moving averages
  - name: "sma_10"
    type: "simple_moving_average"
    period: 10
    
  - name: "ema_21"  
    type: "exponential_moving_average"
    period: 21
    
  # Momentum indicators
  - name: "rsi_14"
    type: "relative_strength_index"
    period: 14
    
  - name: "macd"
    type: "macd"
    fast_period: 12
    slow_period: 26
    signal_period: 9
    
  # Volatility indicators
  - name: "atr_20"
    type: "average_true_range"
    period: 20
    
  - name: "bollinger"
    type: "bollinger_bands"
    period: 20
    std_dev: 2.0
```

### Market Regime Classification

```yaml
# Automatically detect market regimes
regime_detection:
  # Hidden Markov Model classifier
  - name: "hmm_classifier"
    type: "hidden_markov_model"
    n_states: 3
    features: ["returns", "volatility", "volume"]
    regime_labels: ["bull", "bear", "neutral"]
    
  # Pattern-based classifier  
  - name: "pattern_classifier"
    type: "pattern_recognition"
    patterns: ["breakout", "reversal", "continuation"]
    lookback_period: 20
    
  # Volatility regime
  - name: "vol_classifier"
    type: "volatility_regime"
    short_window: 10
    long_window: 60
    thresholds: [0.15, 0.25]  # Low/Medium/High vol
```

### Execution Settings

```yaml
execution:
  # Trading costs
  slippage:
    type: "percentage"
    value: 0.001  # 0.1% slippage
    
  commission:
    type: "per_share"
    amount: 0.005  # $0.005 per share
    
  # Order types
  default_order_type: "market"
  
  # Timing
  market_open_delay: 5  # Wait 5 minutes after market open
  market_close_buffer: 10  # Stop trading 10 minutes before close
```

---

## Multi-Phase Optimization

### Simple Parameter Optimization

```yaml
workflow:
  type: "optimization"
  name: "Find Best Parameters"

# Define what to optimize
optimization:
  algorithm: "grid_search"  # Test all combinations
  objective: "sharpe_ratio" # What to maximize
  
  # Which parameters to test
  parameter_space:
    fast_period: [5, 10, 15, 20]      # 4 values
    slow_period: [20, 30, 40, 50]     # 4 values  
    # Will test 4 × 4 = 16 combinations
    
  # Validation
  validation:
    method: "walk_forward"
    train_pct: 70  # Use 70% for training, 30% for testing
```

### Advanced Multi-Phase Optimization

```yaml
workflow:
  type: "optimization"
  name: "Complete Strategy Optimization"

# Phase 1: Find best individual strategies
optimization:
  phase_1:
    name: "strategy_selection"
    algorithm: "genetic"
    objective: "risk_adjusted_return"
    
    # Test multiple strategy types
    strategies:
      momentum:
        parameter_space:
          fast_period: [5, 10, 15, 20, 25]
          slow_period: [20, 30, 40, 50, 60]
          
      mean_reversion:
        parameter_space:
          lookback_period: [10, 15, 20, 25, 30]
          std_threshold: [1.5, 2.0, 2.5, 3.0]
          
      breakout:
        parameter_space:
          breakout_period: [10, 15, 20, 25]
          volume_threshold: [1.2, 1.5, 2.0, 2.5]
    
    # Genetic algorithm settings
    population_size: 100
    generations: 50
    mutation_rate: 0.1
  
  # Phase 2: Optimize combination of best strategies
  phase_2:
    name: "portfolio_optimization"
    algorithm: "bayesian"
    objective: "calmar_ratio"
    
    # Use top 5 strategies from Phase 1
    input: "phase_1_top_5"
    
    # Optimize weights and risk settings
    parameter_space:
      strategy_weights:
        type: "simplex"  # Weights that sum to 1.0
        min_weight: 0.1  # Minimum 10% allocation
        
      risk_settings:
        position_size_pct: [1.0, 1.5, 2.0, 2.5, 3.0]
        max_drawdown_pct: [10, 15, 20, 25]
  
  # Phase 3: Out-of-sample validation
  phase_3:
    name: "validation"
    
    # Use best portfolio from Phase 2
    input: "phase_2_best"
    
    # Test on fresh data
    validation_data:
      start_date: "2024-01-01"  # Different from training data
      end_date: "2024-12-31"
      
    # Walk-forward analysis
    walk_forward:
      training_window: 252  # 1 year
      test_window: 63      # 3 months
      step_size: 21        # 1 month steps
```

### Optimization Output Control

```yaml
optimization:
  # ... optimization settings ...
  
  # Control output to manage disk space
  output:
    # Only save top performers
    save_top_n: 50
    
    # What to save for each trial
    save_equity_curves: false  # These can be large
    save_trade_logs: false
    save_parameters: true      # Always save parameters
    save_metrics: true         # Always save performance metrics
    
    # Detailed results for top performers only
    detailed_results_top_n: 10
```

---

## Live Trading Configuration

### Paper Trading Setup

```yaml
workflow:
  type: "live_trading"
  name: "Paper Trading Test"

# Start with paper trading
broker:
  name: "alpaca"  # or "interactive_brokers", "td_ameritrade"
  paper_trading: true  # No real money
  api_key: "${ALPACA_API_KEY}"
  secret_key: "${ALPACA_SECRET_KEY}"

# Use strategies optimized in backtesting
strategies:
  - name: "optimized_momentum"
    type: "momentum"
    # Use parameters found in optimization
    fast_period: 12  # From optimization results
    slow_period: 28  # From optimization results
    symbols: ["AAPL", "GOOGL", "MSFT"]

# Conservative risk for live trading
risk:
  initial_capital: 25000  # Start small
  position_size_pct: 1.0  # Half the backtest size
  max_drawdown_pct: 5.0   # Tighter controls
  
  # Real-time risk monitoring
  real_time_limits:
    max_daily_loss: 500
    max_hourly_loss: 100
    
# Monitoring and alerts
monitoring:
  email_alerts: true
  email: "trader@example.com"
  
  slack_alerts: true
  slack_webhook: "${SLACK_WEBHOOK}"
  
  # What triggers alerts
  alert_triggers:
    - "daily_loss_limit"
    - "position_limit_breach" 
    - "system_error"
    - "unusual_market_conditions"
```

### Production Live Trading

```yaml
workflow:
  type: "live_trading"
  name: "Production Strategy"

# Real money trading
broker:
  name: "interactive_brokers"
  paper_trading: false  # REAL MONEY
  account_id: "${IB_ACCOUNT_ID}"
  gateway_host: "127.0.0.1"
  gateway_port: 7497

# Proven strategies only
strategies:
  - name: "validated_momentum"
    type: "momentum"
    # Parameters validated through full optimization pipeline
    fast_period: 12
    slow_period: 28
    symbols: ["AAPL", "GOOGL", "MSFT", "SPY"]

# Production risk management
risk:
  initial_capital: 100000
  
  # Conservative position sizing
  position_sizing:
    type: "volatility_based"
    risk_per_trade: 0.5  # 0.5% risk per trade
    
  # Multiple safety limits
  limits:
    - type: "daily_loss"
      amount: 1000
    - type: "weekly_loss"  
      amount: 3000
    - type: "monthly_loss"
      amount: 8000
    - type: "max_drawdown"
      percent: 8.0
      
  # Emergency stops
  circuit_breakers:
    - type: "consecutive_losses"
      count: 5  # Stop after 5 consecutive losses
    - type: "rapid_loss"
      amount: 2000
      timeframe: "1H"  # $2k loss in 1 hour

# Comprehensive monitoring
monitoring:
  # Real-time dashboard
  dashboard:
    enabled: true
    port: 8080
    auth_token: "${DASHBOARD_TOKEN}"
    
  # Multiple alert channels
  alerts:
    email:
      enabled: true
      recipients: ["trader@company.com", "risk@company.com"]
    
    sms:
      enabled: true
      phone: "${TRADER_PHONE}"
      
    slack:
      enabled: true
      channel: "#trading-alerts"
      
  # Regulatory logging
  audit_log:
    enabled: true
    path: "/var/log/trading/"
    retention_days: 2555  # 7 years
```

---

## Complete Examples

### Example 1: Simple Momentum Backtest

```yaml
# momentum_backtest.yaml
workflow:
  type: "backtest"
  name: "Tech Stock Momentum Strategy"
  description: "Testing momentum on large tech stocks"

data:
  symbols: ["AAPL", "GOOGL", "MSFT", "AMZN"]
  start_date: "2022-01-01"  
  end_date: "2023-12-31"
  timeframe: "1D"

strategies:
  - name: "momentum_strategy"
    type: "momentum"
    fast_period: 10
    slow_period: 30
    signal_threshold: 0.02

risk:
  initial_capital: 100000
  position_size_pct: 5.0
  max_drawdown_pct: 15.0

output:
  path: "results/momentum_backtest/"
  save_trades: true
  save_equity_curve: true
  generate_report: true
```

### Example 2: Multi-Strategy Portfolio

```yaml
# multi_strategy_portfolio.yaml
workflow:
  type: "backtest"
  name: "Diversified Strategy Portfolio"

data:
  symbols: ["AAPL", "GOOGL", "MSFT", "SPY", "QQQ", "IWM", "GLD", "TLT"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  timeframe: "1D"

strategies:
  # Momentum for tech stocks
  - name: "tech_momentum"
    type: "momentum"
    fast_period: 12
    slow_period: 26
    symbols: ["AAPL", "GOOGL", "MSFT"]
    
  # Mean reversion for ETFs
  - name: "etf_mean_reversion"
    type: "mean_reversion"
    lookback_period: 20
    std_threshold: 2.0
    symbols: ["SPY", "QQQ", "IWM"]
    
  # Breakout for commodities
  - name: "commodity_breakout"
    type: "breakout"
    breakout_period: 20
    atr_multiplier: 2.0
    symbols: ["GLD"]
    
  # Pairs trading
  - name: "bond_equity_pairs"
    type: "pairs_trading"
    symbol_1: "SPY"
    symbol_2: "TLT"
    lookback_period: 60
    entry_threshold: 2.0

# Portfolio allocation
strategy_allocation:
  tech_momentum: 0.4
  etf_mean_reversion: 0.3
  commodity_breakout: 0.2
  bond_equity_pairs: 0.1

risk:
  initial_capital: 200000
  position_sizing:
    type: "equal_weight"
  max_portfolio_risk: 25.0
  max_correlation: 0.6
```

### Example 3: Complete Optimization Pipeline

```yaml
# optimization_pipeline.yaml
workflow:
  type: "optimization"
  name: "Complete Strategy Development Pipeline"

data:
  symbols: ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
  start_date: "2018-01-01"
  end_date: "2023-06-30"  # Training data
  timeframe: "1D"

# Validation data (out-of-sample)
validation_data:
  start_date: "2023-07-01"
  end_date: "2023-12-31"

optimization:
  # Phase 1: Individual strategy optimization
  phase_1:
    name: "parameter_search"
    algorithm: "grid_search"
    objective: "sharpe_ratio"
    
    strategies:
      momentum:
        parameter_space:
          fast_period: [5, 8, 10, 12, 15]
          slow_period: [20, 26, 30, 35, 40]
          signal_threshold: [0.01, 0.015, 0.02, 0.025]
      
      mean_reversion:
        parameter_space:
          lookback_period: [15, 20, 25, 30]
          std_threshold: [1.5, 2.0, 2.5]
          hold_period: [3, 5, 7, 10]
    
    # Test different market regimes
    regime_classifiers:
      - name: "hmm_3_state"
        type: "hidden_markov_model"
        n_states: 3
      - name: "volatility_regime"
        type: "volatility_classification"
        
    # Test different risk profiles
    risk_profiles:
      - name: "conservative"
        position_size_pct: 1.0
        max_exposure: 10.0
      - name: "moderate" 
        position_size_pct: 2.0
        max_exposure: 20.0
        
  # Phase 2: Portfolio optimization
  phase_2:
    name: "portfolio_construction"
    algorithm: "genetic"
    objective: "calmar_ratio"
    
    # Use top 10 strategy configurations from Phase 1
    input: "phase_1_top_10"
    
    # Optimize allocation weights
    population_size: 100
    generations: 200
    
  # Phase 3: Walk-forward validation
  phase_3:
    name: "robustness_testing"
    
    input: "phase_2_best"
    
    walk_forward:
      training_window: 504  # 2 years
      test_window: 126     # 6 months
      step_size: 21        # 1 month

output:
  base_path: "results/optimization_pipeline/"
  
  phase_1_output:
    save_top_n: 100
    generate_heatmaps: true
    
  phase_2_output:
    save_population_evolution: true
    save_correlation_analysis: true
    
  phase_3_output:
    save_period_breakdown: true
    generate_final_report: true
```

---

## Environment Variables and Security

### Using Environment Variables

For sensitive information like API keys:

```yaml
# config.yaml
broker:
  api_key: "${BROKER_API_KEY}"
  secret: "${BROKER_SECRET}"
  
database:
  connection: "${DATABASE_URL}"
  
notifications:
  slack_webhook: "${SLACK_WEBHOOK}"
  email_password: "${EMAIL_PASSWORD}"
```

Set environment variables:
```bash
export BROKER_API_KEY="your_api_key_here"
export BROKER_SECRET="your_secret_here"
export DATABASE_URL="postgresql://user:pass@localhost/db"
```

### Configuration Inheritance

Create base configurations and extend them:

```yaml
# base_config.yaml
defaults: &defaults
  data:
    timeframe: "1D"
    preprocessing:
      fill_missing: "forward_fill"
      
  execution:
    slippage: {type: "percentage", value: 0.001}
    commission: {type: "fixed", amount: 1.0}

# production_config.yaml
<<: *defaults  # Inherit base settings

workflow:
  type: "live_trading"
  name: "Production Strategy"

# Override specific settings
broker:
  paper_trading: false
  
risk:
  # More conservative for live trading
  position_size_pct: 1.0  # Override default
```

---

## Troubleshooting

### Common Configuration Errors

#### 1. Invalid Date Format
```yaml
# ❌ Wrong
data:
  start_date: "01/01/2023"  # MM/DD/YYYY not supported
  
# ✅ Correct  
data:
  start_date: "2023-01-01"  # YYYY-MM-DD format
```

#### 2. Missing Required Fields
```yaml
# ❌ Wrong - missing workflow type
data:
  symbols: ["AAPL"]
  
# ✅ Correct
workflow:
  type: "backtest"  # Always specify workflow type
data:
  symbols: ["AAPL"]
```

#### 3. Invalid Parameter Ranges
```yaml
# ❌ Wrong - slow period must be > fast period
strategies:
  - type: "momentum"
    fast_period: 30
    slow_period: 10  # Invalid: slow < fast
    
# ✅ Correct
strategies:
  - type: "momentum"
    fast_period: 10
    slow_period: 30
```

### Validation

ADMF-PC validates your configuration before running:

```bash
# Check configuration without running
admf-validate my_config.yaml

# Output:
✓ Configuration valid
ℹ Using 252 trading days
ℹ Testing 4 strategies across 2 symbols
⚠ Warning: Large parameter space (1000+ combinations)
```

### Getting Help

```bash
# Show available strategy types
admf-help strategies

# Show parameters for specific strategy
admf-help strategy momentum

# Show example configurations
admf-examples

# Validate and explain configuration
admf-explain my_config.yaml
```

---

## Next Steps

1. **Start Simple**: Begin with the basic backtest example
2. **Add Complexity Gradually**: Add more strategies, risk controls, etc.
3. **Optimize**: Use parameter optimization to find best settings
4. **Validate**: Test on out-of-sample data
5. **Paper Trade**: Deploy to paper trading first
6. **Go Live**: Move to production with proven strategies

### Learning Path

1. **Week 1**: Basic backtests with single strategies
2. **Week 2**: Multi-strategy portfolios and risk management
3. **Week 3**: Parameter optimization
4. **Week 4**: Multi-phase optimization pipelines
5. **Week 5**: Paper trading deployment
6. **Week 6+**: Live trading with full monitoring

Remember: ADMF-PC handles all the complex programming. You just need to describe what you want to test in YAML. The system takes care of the rest!
