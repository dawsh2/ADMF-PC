# Welcome to ADMF-PC: Complete Onboarding Guide

## 🎯 Goal: Become Productive in 2 Hours

This guide will take you from zero to building sophisticated trading strategies with ADMF-PC. By the end, you'll understand not just *how* to use the system, but *why* it's designed this way.

## 📋 Table of Contents

1. [What is ADMF-PC?](#what-is-admf-pc)
2. [Why Protocol + Composition?](#why-protocol-composition)
3. [Your First Hour](#your-first-hour)
4. [Core Architecture](#core-architecture)
5. [Building Trading Strategies](#building-trading-strategies)
6. [Advanced Features](#advanced-features)
7. [Next Steps](#next-steps)

---

## What is ADMF-PC?

ADMF-PC (Adaptive Decision Making Framework - Protocol Composition) is a **zero-code algorithmic trading system** that revolutionizes how trading strategies are built and deployed.

### Traditional Approach vs ADMF-PC

#### Traditional (What You DON'T Need to Do)
```python
class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.fast_ma = MovingAverage(10)
        self.slow_ma = MovingAverage(30)
        
    def on_bar(self, bar):
        if self.fast_ma.value > self.slow_ma.value:
            self.buy(bar.symbol, 100)
        # ... hundreds more lines
```

#### ADMF-PC (What You Actually Do)
```yaml
strategies:
  - type: momentum
    fast_period: 10
    slow_period: 30
```

**That's it!** The system handles everything else.

### Key Benefits

1. **Zero Code Required**: Pure configuration
2. **Instant Experiments**: Change parameters, run again
3. **Mix Everything**: Any indicator, any strategy, any data source
4. **Production Ready**: Same config for backtest and live trading
5. **Institutional Scale**: Handles billions in AUM

---

## Why Protocol + Composition?

### The Problem with Inheritance

Traditional frameworks force you into rigid hierarchies:

```
BaseStrategy
  ├── TrendStrategy
  │   ├── MovingAverageStrategy  
  │   └── MomentumStrategy     ← Can't mix with MeanReversion!
  └── MeanReversionStrategy
      └── BollingerStrategy    ← Locked in this tree
```

### The ADMF-PC Solution

Everything is composable through protocols:

```yaml
strategies:
  - type: ensemble
    components:
      - type: momentum           # From one "tree"
      - type: mean_reversion     # From another "tree"  
      - function: my_custom_func # Your own function
      - ml_model: sklearn.RandomForest  # External library
```

**You can mix ANYTHING because there are no inheritance constraints!**

---

## Your First Hour

Let's build understanding through hands-on examples.

### ✅ Task 1: Run a Simple Backtest (10 min)

1. Create `simple_backtest.yaml`:

```yaml
workflow:
  type: "backtest"
  
data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  
strategies:
  - type: "momentum"
    fast_period: 10
    slow_period: 30
    
risk:
  initial_capital: 100000
  position_size_pct: 2.0
```

2. Run it:
```bash
python main.py simple_backtest.yaml
```

3. Check results in `output/backtest_*/performance_report.html`

### ✅ Task 2: Add Risk Management (10 min)

Enhance your strategy with risk controls:

```yaml
risk:
  initial_capital: 100000
  position_size_pct: 2.0
  max_drawdown_pct: 10.0      # Stop trading if down 10%
  stop_loss_pct: 2.0          # Exit if position down 2%
  take_profit_pct: 5.0        # Exit if position up 5%
  max_positions: 5            # Limit concurrent positions
```

### ✅ Task 3: Combine Multiple Strategies (15 min)

```yaml
strategies:
  # Trend follower
  - name: "trend"
    type: "momentum"
    fast_period: 10
    slow_period: 30
    allocation: 0.4
    
  # Mean reverter  
  - name: "reverter"
    type: "mean_reversion"
    lookback_period: 20
    entry_threshold: 2.0
    allocation: 0.3
    
  # Volatility trader
  - name: "vol_trader"
    type: "volatility_breakout"
    atr_period: 14
    breakout_multiplier: 1.5
    allocation: 0.3
```

### ✅ Task 4: Add Market Regime Detection (15 min)

Make your strategies adapt to market conditions:

```yaml
# Define regime classifier
classifiers:
  - name: "market_regime"
    type: "hmm"  # Hidden Markov Model
    states: ["bull", "bear", "neutral"]
    features: ["returns", "volatility"]
    
# Regime-aware strategy
strategies:
  - name: "adaptive"
    type: "regime_adaptive"
    classifiers: ["market_regime"]
    regime_strategies:
      bull:
        type: "momentum"
        fast_period: 5
        slow_period: 15
      bear:
        type: "mean_reversion"
        lookback_period: 20
      neutral:
        type: "market_neutral"
        hedge_ratio: 0.8
```

### ✅ Task 5: Run Parameter Optimization (10 min)

Find the best parameters automatically:

```yaml
workflow:
  type: "optimization"
  algorithm: "grid_search"
  objective: "sharpe_ratio"
  
parameter_space:
  fast_period: [5, 10, 15, 20]
  slow_period: [20, 30, 40, 50]
  signal_threshold: [0.01, 0.02, 0.03]
  
validation:
  method: "walk_forward"
  train_periods: 252  # 1 year
  test_periods: 63    # 3 months
  step_size: 21       # 1 month
```

---

## Core Architecture

Understanding the architecture helps you leverage ADMF-PC's full power.

### Event-Driven Flow

```
Market Data → Indicators → Strategies → Signals → Risk → Orders → Execution
     ↓            ↓           ↓           ↓        ↓       ↓         ↓
   [BAR]    [INDICATOR]  [STRATEGY]  [SIGNAL]  [ORDER] [FILL] [POSITION]
```

Everything communicates through events, making the system:
- **Reactive**: Components respond to relevant events
- **Scalable**: Add more components without changing others
- **Testable**: Mock any component easily

### Container Architecture

```
System Container
  ├── Data Container
  │   └── Handles all data feeds
  ├── Strategy Container  
  │   ├── Strategy A (isolated)
  │   └── Strategy B (isolated)
  ├── Risk Container
  │   └── Enforces all limits
  └── Execution Container
      └── Manages orders
```

Each container:
- **Isolated**: No shared state
- **Composable**: Mix and match
- **Replaceable**: Swap implementations

### Configuration-Driven Design

Your YAML configuration becomes the entire system specification:

```yaml
# This configuration fully defines:
# - What data to use
# - Which strategies to run  
# - How to manage risk
# - Where to save results

workflow:
  type: "backtest"
  
# ... rest of config
```

---

## Building Trading Strategies

### Available Strategy Types

#### Trend Following
```yaml
strategies:
  - type: "momentum"
    fast_period: 10
    slow_period: 30
    
  - type: "breakout"
    lookback_period: 20
    breakout_factor: 2.0
```

#### Mean Reversion
```yaml
strategies:
  - type: "mean_reversion"
    lookback_period: 20
    entry_threshold: 2.0
    exit_threshold: 0.5
    
  - type: "pairs_trading"
    symbol_1: "AAPL"
    symbol_2: "MSFT"
    lookback: 60
```

#### Advanced Strategies
```yaml
strategies:
  - type: "ml_ensemble"
    models:
      - type: "random_forest"
        features: ["rsi", "macd", "volume_ratio"]
      - type: "neural_network"
        layers: [50, 30, 10]
      - type: "xgboost"
        max_depth: 5
```

### Custom Indicators

Add any indicator through configuration:

```yaml
indicators:
  - name: "custom_momentum"
    formula: "(close - close[20]) / close[20]"
    
  - name: "volume_spike"
    formula: "volume > volume.rolling(20).mean() * 2"
    
strategies:
  - type: "indicator_based"
    buy_condition: "custom_momentum > 0.05 and volume_spike"
    sell_condition: "custom_momentum < -0.02"
```

### Risk Management Options

```yaml
risk:
  # Position sizing
  position_sizing_method: "volatility_adjusted"
  risk_per_trade: 0.01  # 1% risk per trade
  
  # Portfolio limits
  max_portfolio_heat: 0.06  # 6% total risk
  max_correlation: 0.7       # Limit correlated positions
  
  # Drawdown control
  max_drawdown_pct: 15.0
  drawdown_halflife_days: 20  # Gradual recovery
  
  # Execution limits
  max_slippage_pct: 0.1
  max_spread_pct: 0.05
```

---

## Advanced Features

### Multi-Phase Workflows

Chain operations for sophisticated research:

```yaml
workflow:
  type: "multi_phase"
  
phases:
  - name: "parameter_discovery"
    type: "optimization"
    objective: "sharpe_ratio"
    
  - name: "regime_analysis"
    type: "analysis"
    analyze_performance_by: ["market_regime", "volatility_regime"]
    
  - name: "portfolio_construction"
    type: "optimization"
    objective: "risk_adjusted_return"
    constraints: ["max_correlation", "min_sharpe"]
    
  - name: "out_of_sample_test"
    type: "backtest"
    use_optimized_parameters: true
```

### Walk-Forward Analysis

Robust testing with rolling windows:

```yaml
validation:
  method: "walk_forward"
  windows:
    - train_start: "2020-01-01"
      train_end: "2020-12-31"
      test_start: "2021-01-01"
      test_end: "2021-03-31"
    - train_start: "2020-04-01"
      train_end: "2021-03-31"
      test_start: "2021-04-01"
      test_end: "2021-06-30"
  reoptimize_each_window: true
```

### Signal Analysis

Understand your strategy's behavior:

```yaml
analysis:
  signal_analysis:
    - mae_mfe  # Maximum adverse/favorable excursion
    - signal_correlation
    - regime_performance
    - time_of_day_analysis
    
  save_artifacts:
    - all_signals
    - indicator_values
    - regime_states
```

---

## Next Steps

### 📚 Essential Reading

1. **[Key Concepts](CONCEPTS.md)** - Deep dive into architecture
2. **[Common Pitfalls](COMMON_PITFALLS.md)** - Avoid common mistakes
3. **[Component Catalog](../COMPONENT_CATALOG.md)** - All available components

### 🎯 Learning Paths

Choose based on your goals:

1. **[Strategy Developer Path](learning-paths/strategy-developer.md)**
   - Create custom strategies
   - Optimize parameters
   - Analyze performance

2. **[System Integrator Path](learning-paths/system-integrator.md)**
   - Connect data sources
   - Deploy to production
   - Monitor systems

3. **[Researcher Path](learning-paths/researcher.md)**
   - Run experiments
   - Analyze market regimes
   - Develop new models

4. **[ML Practitioner Path](learning-paths/ml-practitioner.md)**
   - Integrate ML models
   - Feature engineering
   - Model validation

### 🚀 Ready for Production?

When you're ready to go live:

1. **Test Thoroughly**: Use walk-forward validation
2. **Start Small**: Begin with small capital
3. **Monitor Closely**: Set up alerts and dashboards
4. **Scale Gradually**: Increase size as confidence grows

### 🎓 Continue Learning

- **[Complexity Guide](../complexity-guide/README.md)** - Advanced features step-by-step
- **[System Architecture](../SYSTEM_ARCHITECTURE_V5.MD)** - Technical deep dive
- **[Style Guide](../standards/STYLE-GUIDE.md)** - Coding standards and best practices
- **[GitHub Repository](#)** - Latest updates and examples

---

## 🎉 Congratulations!

You now understand:
- ✅ How to configure strategies without code
- ✅ The Protocol + Composition philosophy
- ✅ Container architecture benefits
- ✅ How to combine multiple strategies
- ✅ Risk management configuration
- ✅ How to run optimizations

**You're ready to build sophisticated trading systems with ADMF-PC!**

---

*Questions? Check [FAQ.md](FAQ.md) | Need help? See [RESOURCES.md](RESOURCES.md)*

[← Back to Hub](README.md) | [Next: First Task →](FIRST_TASK.md)