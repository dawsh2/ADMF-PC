# Learning Path: Researcher

A comprehensive guide for quantitative researchers using ADMF-PC for strategy research, backtesting, and optimization.

## Overview

As a Researcher, you'll learn how to:
- Rapidly test trading hypotheses
- Run sophisticated backtests
- Optimize strategy parameters
- Analyze regime-specific performance
- Validate results statistically

## Prerequisites

- Understanding of financial markets
- Basic statistics knowledge
- Familiarity with trading concepts
- No programming required!

## Learning Path

### Phase 1: Research Fundamentals (Week 1)

#### Day 1-2: System Basics
- [ ] Complete [Quick Start Guide](../QUICK_START.md)
- [ ] Run your first backtest
- [ ] Understand YAML configuration
- [ ] Explore example strategies

#### Day 3-4: Core Concepts
- [ ] Study [Three-Pattern Architecture](../../architecture/04-THREE-PATTERN-BACKTEST.md)
- [ ] Learn about Signal Generation vs Execution
- [ ] Understand event flow

#### Day 5-7: Research Tools
- [ ] Master configuration options
- [ ] Learn performance metrics
- [ ] Understand validation methods

### Phase 2: Strategy Research (Week 2)

#### Testing Trading Ideas

**Example 1: Momentum Strategy Research**
```yaml
# research_momentum.yaml
workflow:
  type: "signal_generation"  # Just analyze signals
  name: "Momentum Signal Quality Research"

data:
  symbols: ["SPY", "QQQ", "IWM"]
  start_date: "2020-01-01"
  end_date: "2023-12-31"

strategies:
  - name: "fast_momentum"
    type: "momentum"
    fast_period: 5
    slow_period: 20
    
  - name: "medium_momentum"
    type: "momentum"
    fast_period: 10
    slow_period: 50
    
  - name: "slow_momentum"
    type: "momentum"
    fast_period: 20
    slow_period: 100

analysis:
  metrics:
    - signal_accuracy
    - average_signal_duration
    - signal_stability
    - regime_performance
```

**Example 2: Mean Reversion Research**
```yaml
# research_mean_reversion.yaml
strategies:
  - name: "bollinger_reversion"
    type: "mean_reversion"
    lookback: 20
    num_std: 2.0
    
  - name: "rsi_reversion"
    type: "mean_reversion"
    indicator: "rsi"
    oversold: 30
    overbought: 70

analysis:
  regime_splits:
    - high_volatility
    - low_volatility
    - trending
    - ranging
```

#### Research Tasks:
- [ ] Test 5 different momentum periods
- [ ] Compare signal quality metrics
- [ ] Analyze regime-specific performance
- [ ] Document findings

### Phase 3: Parameter Optimization (Week 3)

#### Grid Search Optimization
```yaml
# optimization_grid_search.yaml
workflow:
  type: "optimization"
  algorithm: "grid_search"

parameter_space:
  strategies.momentum.fast_period: [5, 10, 15, 20]
  strategies.momentum.slow_period: [20, 30, 50, 100]
  risk.position_size_pct: [1.0, 2.0, 3.0]

optimization:
  objective: "sharpe_ratio"
  constraints:
    max_drawdown: 0.20
    min_trades: 50
```

#### Adaptive Optimization
```yaml
# optimization_adaptive.yaml
workflow:
  type: "optimization"
  algorithm: "adaptive"

phases:
  - name: "coarse_search"
    resolution: 0.2  # 20% steps
    
  - name: "fine_search"  
    resolution: 0.05  # 5% steps
    search_around: "best_from_previous"
    
  - name: "final_tuning"
    resolution: 0.01  # 1% steps
    candidates: 10  # Top 10 from previous
```

#### Walk-Forward Analysis
```yaml
# walk_forward_validation.yaml
workflow:
  type: "walk_forward"
  
walk_forward:
  training_period: 252  # 1 year
  test_period: 63      # 3 months
  step_size: 21        # 1 month
  
  optimization_method: "grid_search"
  reoptimize_frequency: "monthly"
```

### Phase 4: Advanced Research (Week 4+)

#### Multi-Strategy Portfolios
```yaml
# portfolio_research.yaml
strategies:
  - name: "trend_following"
    type: "momentum"
    allocation: 0.4
    
  - name: "mean_reversion"
    type: "mean_reversion"
    allocation: 0.3
    
  - name: "volatility_arbitrage"
    type: "volatility"
    allocation: 0.3

portfolio:
  rebalance_frequency: "weekly"
  correlation_limit: 0.7
  risk_parity: true
```

#### Regime-Aware Strategies
```yaml
# regime_adaptive.yaml
classifiers:
  - name: "market_regime"
    type: "hmm"
    states: ["bull", "bear", "sideways"]
    features: ["returns", "volatility", "volume"]

strategies:
  bull_market:
    type: "momentum"
    aggressiveness: "high"
    
  bear_market:
    type: "defensive"
    cash_allocation: 0.5
    
  sideways_market:
    type: "mean_reversion"
    frequency: "high"
```

## Research Workflows

### 1. Hypothesis Testing Workflow

```yaml
# Step 1: Generate signals
workflow:
  type: "signal_generation"
  capture_signals: true
  
# Step 2: Analyze signal quality
analysis:
  statistical_tests:
    - t_test_vs_random
    - sharpe_ratio_significance
    - information_ratio
    
# Step 3: Validate out-of-sample
validation:
  method: "expanding_window"
  min_training_periods: 252
```

### 2. Strategy Development Workflow

1. **Idea Generation**
   - Market observation
   - Academic research
   - Anomaly detection

2. **Initial Testing**
```yaml
# Quick test with one symbol
data:
  symbols: ["SPY"]
  start_date: "2022-01-01"
  end_date: "2022-12-31"
```

3. **Expanded Testing**
```yaml
# Test across multiple symbols and timeframes
data:
  symbols: ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"]
  timeframes: ["1D", "4H", "1H"]
```

4. **Optimization**
```yaml
# Find optimal parameters
optimization:
  method: "bayesian"
  n_iterations: 100
```

5. **Validation**
```yaml
# Out-of-sample testing
validation:
  test_set_size: 0.3
  methods: ["walk_forward", "monte_carlo"]
```

### 3. Performance Analysis Workflow

#### Comprehensive Metrics
```yaml
analysis:
  performance_metrics:
    - total_return
    - sharpe_ratio
    - sortino_ratio
    - max_drawdown
    - calmar_ratio
    - win_rate
    - profit_factor
    
  risk_metrics:
    - value_at_risk
    - conditional_value_at_risk
    - downside_deviation
    - ulcer_index
    
  trade_metrics:
    - average_trade_duration
    - trades_per_day
    - average_win_loss_ratio
    - consecutive_wins_losses
```

#### Attribution Analysis
```yaml
attribution:
  by_symbol: true
  by_time_period: true
  by_market_regime: true
  by_signal_strength: true
```

## Research Best Practices

### 1. Avoid Overfitting

```yaml
# Good: Reasonable parameter ranges
parameter_space:
  lookback: [10, 20, 30, 50]  # Sensible values

# Bad: Too granular
parameter_space:
  lookback: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Overfitting risk
```

### 2. Use Proper Validation

```yaml
# Always reserve out-of-sample data
data_split:
  training: 0.6
  validation: 0.2
  test: 0.2  # Never touch until final validation
```

### 3. Consider Transaction Costs

```yaml
execution:
  costs:
    commission: 0.001  # $1 per 1000 shares
    slippage_model: "percentage"
    slippage_pct: 0.05
    market_impact: "square_root"
```

### 4. Test Robustness

```yaml
robustness_tests:
  parameter_stability:
    perturbation: 0.1  # Test ±10% parameter changes
    
  data_sensitivity:
    methods:
      - bootstrap
      - jackknife
      - monte_carlo
      
  regime_stability:
    test_regimes:
      - high_volatility
      - low_volatility
      - crisis_periods
```

## Statistical Validation

### Significance Testing
```python
# Built-in statistical tests
validation:
  statistical_tests:
    sharpe_ratio_test:
      null_hypothesis: 0  # Random walk
      confidence_level: 0.95
      
    information_ratio_test:
      benchmark: "SPY"
      min_threshold: 0.5
```

### Monte Carlo Simulation
```yaml
monte_carlo:
  simulations: 1000
  methods:
    - random_entry_exit
    - parameter_perturbation
    - data_resampling
    
  confidence_intervals: [0.05, 0.25, 0.75, 0.95]
```

## Research Tools

### 1. Signal Quality Analyzer
```yaml
# Analyze signal characteristics
signal_analysis:
  metrics:
    - signal_frequency
    - signal_clustering
    - signal_autocorrelation
    - signal_to_noise_ratio
    
  visualizations:
    - signal_distribution
    - signal_heatmap
    - regime_performance
```

### 2. Strategy Comparator
```yaml
# Compare multiple strategies
comparison:
  strategies: ["momentum", "mean_reversion", "pairs_trading"]
  
  metrics:
    - risk_adjusted_returns
    - correlation_matrix
    - drawdown_synchronicity
    - diversification_benefit
```

### 3. Regime Analyzer
```yaml
# Understand strategy behavior in different regimes
regime_analysis:
  regime_detection: "hmm_3_state"
  
  performance_by_regime: true
  transition_analysis: true
  regime_persistence: true
```

## Common Research Patterns

### 1. Momentum Research
```yaml
# Test momentum across timeframes
research_patterns:
  momentum_term_structure:
    periods: [5, 10, 20, 60, 120, 252]
    holding_periods: [1, 5, 20, 60]
    
  momentum_decay:
    signal_age_buckets: [0-5, 5-10, 10-20, 20+]
```

### 2. Mean Reversion Research
```yaml
# Test mean reversion conditions
research_patterns:
  reversion_conditions:
    volatility_regimes: ["low", "medium", "high"]
    deviation_thresholds: [1.5, 2.0, 2.5, 3.0]
    holding_periods: [1, 2, 5, 10]
```

### 3. Factor Research
```yaml
# Multi-factor analysis
factors:
  - name: "value"
    metric: "book_to_market"
  - name: "momentum"
    metric: "12_month_return"
  - name: "quality"
    metric: "roe"
  - name: "low_volatility"
    metric: "realized_vol"

factor_analysis:
  method: "cross_sectional_regression"
  rebalance: "monthly"
```

## Publishing Research

### Research Report Template
```yaml
report:
  sections:
    - executive_summary
    - methodology
    - backtest_results
    - statistical_validation
    - risk_analysis
    - implementation_guide
    
  visualizations:
    - equity_curves
    - drawdown_chart
    - monthly_returns_heatmap
    - rolling_metrics
    
  export_formats:
    - pdf
    - html
    - jupyter_notebook
```

## Next Steps

After completing this learning path:
1. Develop your own trading strategies
2. Build a strategy research library
3. Create custom analysis tools
4. Share research with the community

## Resources

- [Signal Analysis Guide](../../complexity-guide/03-signal-capture-replay/README.md)
- [Optimization Patterns](../../complexity-guide/04-multi-phase-integration/README.md)
- [Statistical Validation](../../complexity-guide/validation-framework/README.md)
- [Example Strategies](../../strategy/strategies/)

---

*Remember: Good research is repeatable, validated, and robust across different market conditions. Always validate out-of-sample and consider transaction costs.*