# Optimization

Parameter optimization finds the best configuration for your trading strategies. ADMF-PC provides multiple optimization methods with signal replay for 10-100x speedup.

## 🎯 Basic Optimization

### Simple Grid Search

```yaml
# config/basic_optimization.yaml
workflow:
  type: "optimization"
  name: "Basic Parameter Optimization"
  
data:
  symbols: ["SPY"]
  start_date: "2022-01-01"
  end_date: "2023-12-31"
  
strategies:
  - type: "momentum"
    optimization_target: true        # Mark strategy for optimization
    
optimization:
  method: "grid"
  
  # Parameters to optimize
  parameters:
    fast_period: [5, 10, 15, 20]
    slow_period: [20, 30, 40, 50] 
    signal_threshold: [0.01, 0.02, 0.05]
    
  # Optimization objective
  objective: "sharpe_ratio"          # or "total_return", "calmar_ratio"
  
  # Resource limits
  max_workers: 8                     # Parallel workers
  timeout_hours: 2                   # Maximum runtime
  
reporting:
  output_path: "reports/optimization_results.html"
  include_heatmaps: true
```

### Run the Optimization

```bash
# Run optimization
python main.py config/basic_optimization.yaml

# Run with specific resource limits
python main.py config/basic_optimization.yaml --max-workers 16 --timeout 4h
```

## 🔥 Signal Replay Optimization (10-100x Faster)

### Two-Phase Optimization

```yaml
# Phase 1: Generate signals (slower but necessary)
workflow:
  type: "multi_phase"
  
  phases:
    - name: "signal_generation"
      type: "signal_generation"
      config:
        data:
          symbols: ["SPY"]
          start_date: "2022-01-01"
          end_date: "2023-12-31"
          
        strategies:
          - type: "momentum"
            # Use base parameters for signal generation
            params:
              fast_period: 10
              slow_period: 20
              
        output:
          signal_path: "signals/momentum_signals.pkl"
          
    # Phase 2: Fast optimization using captured signals
    - name: "parameter_optimization"
      type: "optimization"
      method: "signal_replay"              # 10-100x faster!
      inputs: ["signal_generation.signals"]
      
      config:
        optimization:
          method: "grid"
          parameters:
            signal_threshold: [0.005, 0.01, 0.02, 0.05, 0.1]
            stop_loss_pct: [0.01, 0.015, 0.02, 0.025, 0.03]
            take_profit_pct: [0.03, 0.05, 0.075, 0.1]
            position_size_pct: [0.01, 0.02, 0.03, 0.05]
            
          objective: "sharpe_ratio"
          max_workers: 32                  # Can use more workers
```

**Performance Comparison**:
- Full backtest: 100 parameter combinations = 2 hours
- Signal replay: 100 parameter combinations = 2 minutes (60x faster!)

## 🎛️ Optimization Methods

### 1. Grid Search

```yaml
optimization:
  method: "grid"
  
  parameters:
    fast_period: [5, 10, 15, 20, 25]
    slow_period: [20, 30, 40, 50, 60]
    # Total combinations: 5 × 5 = 25
    
  # Grid search options
  grid_options:
    exhaustive: true                     # Test all combinations
    early_stopping: false               # Complete full grid
```

**Best for**: Small parameter spaces, when you want to see all combinations

### 2. Random Search

```yaml
optimization:
  method: "random"
  n_trials: 1000                        # Number of random trials
  
  parameters:
    fast_period: [1, 50]                # Range for uniform sampling
    slow_period: [10, 200]
    signal_threshold: [0.001, 0.1]
    
  # Random search options
  random_options:
    seed: 42                            # For reproducibility
    sampling: "uniform"                 # or "log_uniform"
```

**Best for**: Large parameter spaces, initial exploration, time-constrained optimization

### 3. Bayesian Optimization

```yaml
optimization:
  method: "bayesian"
  n_trials: 200
  
  parameters:
    fast_period: [1, 50]
    slow_period: [10, 200]
    signal_threshold: [0.001, 0.1]
    
  bayesian_options:
    acquisition_function: "ei"          # Expected improvement
    n_initial_points: 20                # Random points to start
    kernel: "matern"                    # GP kernel
    alpha: 1e-6                         # Noise level
```

**Best for**: Expensive function evaluations, continuous parameters, when you want sample efficiency

### 4. Genetic Algorithm

```yaml
optimization:
  method: "genetic"
  
  genetic_options:
    population_size: 100
    generations: 50
    mutation_rate: 0.1
    crossover_rate: 0.8
    selection_method: "tournament"
    
  parameters:
    fast_period: [1, 50]
    slow_period: [10, 200]
    signal_threshold: [0.001, 0.1]
```

**Best for**: Complex parameter interactions, non-convex optimization landscapes

## 🎯 Optimization Objectives

### Single Objective Optimization

```yaml
optimization:
  objective: "sharpe_ratio"            # Single objective

# Available objectives:
objectives:
  return_based:
    - "total_return"
    - "annualized_return"
    - "excess_return"
    
  risk_adjusted:
    - "sharpe_ratio"                   # Most common
    - "sortino_ratio"
    - "calmar_ratio"
    - "omega_ratio"
    
  risk_based:
    - "max_drawdown"                   # Minimize (use negative)
    - "volatility"                     # Minimize
    - "var_95"                         # Value at Risk
    
  trading_based:
    - "profit_factor"
    - "win_rate"
    - "avg_trade_return"
```

### Multi-Objective Optimization

```yaml
optimization:
  method: "multi_objective"
  
  objectives:
    - name: "sharpe_ratio"
      weight: 0.6                      # 60% weight
      direction: "maximize"
      
    - name: "max_drawdown"
      weight: 0.4                      # 40% weight  
      direction: "minimize"
      
  # Pareto optimization
  pareto_options:
    n_points: 100                      # Points on Pareto frontier
    diversity_preference: 0.1          # Prefer diverse solutions
```

### Custom Objective Functions

```yaml
optimization:
  objective: "custom"
  objective_function: "objectives.custom.my_objective"
  
  # Custom parameters
  objective_params:
    risk_free_rate: 0.02
    benchmark_return: 0.08
    penalty_factor: 0.1
```

**Custom Objective Example**:
```python
# File: objectives/custom.py
def my_objective(backtest_results, risk_free_rate=0.02, benchmark_return=0.08, penalty_factor=0.1):
    """
    Custom objective function
    
    Args:
        backtest_results: BacktestResults object with performance metrics
        risk_free_rate: Risk-free rate for calculations
        benchmark_return: Benchmark return for comparison
        penalty_factor: Penalty for high turnover
        
    Returns:
        float: Objective value (higher is better)
    """
    # Extract metrics
    returns = backtest_results.total_return
    volatility = backtest_results.volatility
    max_dd = backtest_results.max_drawdown
    turnover = backtest_results.turnover
    
    # Calculate risk-adjusted return
    excess_return = returns - risk_free_rate
    sharpe = excess_return / volatility
    
    # Add benchmark comparison
    alpha = returns - benchmark_return
    
    # Penalty for high turnover
    turnover_penalty = penalty_factor * turnover
    
    # Combined objective
    objective = sharpe + alpha - turnover_penalty - max_dd
    
    return objective
```

## 🔍 Advanced Optimization

### Constraint Optimization

```yaml
optimization:
  method: "constrained_bayesian"
  
  parameters:
    fast_period: [1, 50]
    slow_period: [10, 200]
    position_size: [0.01, 0.1]
    
  # Constraints
  constraints:
    # Parameter constraints
    - type: "parameter"
      constraint: "slow_period > fast_period"
      
    # Performance constraints  
    - type: "performance"
      metric: "max_drawdown"
      operator: "<"
      value: 0.2                       # Max 20% drawdown
      
    - type: "performance"
      metric: "total_trades"
      operator: ">"
      value: 50                        # Minimum 50 trades
      
    # Risk constraints
    - type: "risk"
      metric: "volatility"
      operator: "<"
      value: 0.25                      # Max 25% volatility
```

### Regime-Aware Optimization

```yaml
optimization:
  method: "regime_aware"
  
  # First detect regimes
  regime_detection:
    method: "hidden_markov"
    n_regimes: 3
    features: ["volatility", "trend", "volume"]
    
  # Then optimize per regime
  regime_optimization:
    optimize_per_regime: true
    
    parameters:
      # Different parameter ranges per regime
      bull_market:
        fast_period: [5, 15]
        slow_period: [15, 30]
        
      bear_market:
        fast_period: [10, 25]  
        slow_period: [25, 50]
        
      sideways_market:
        fast_period: [15, 30]
        slow_period: [30, 60]
```

### Ensemble Optimization

```yaml
optimization:
  method: "ensemble"
  
  # Optimize individual strategies first
  individual_optimization:
    strategy_1:
      type: "momentum"
      parameters:
        fast_period: [5, 20]
        slow_period: [20, 50]
        
    strategy_2:
      type: "mean_reversion"
      parameters:
        period: [10, 30]
        threshold: [1.5, 3.0]
        
  # Then optimize ensemble weights
  ensemble_optimization:
    method: "signal_replay"            # Fast ensemble optimization
    objective: "ensemble_sharpe"
    
    constraints:
      min_weight: 0.1                  # Minimum 10% per strategy
      max_weight: 0.6                  # Maximum 60% per strategy
      sum_weights: 1.0                 # Weights sum to 100%
```

## 📊 Optimization Monitoring

### Real-Time Progress Tracking

```yaml
optimization:
  monitoring:
    enabled: true
    update_frequency: 10              # Update every 10 trials
    
    # Progress reporting
    progress_metrics:
      - "trials_completed"
      - "best_objective_so_far"
      - "time_elapsed"
      - "estimated_time_remaining"
      
    # Live visualization
    live_plots:
      - "objective_vs_trial"
      - "parameter_importance"
      - "convergence_plot"
      
    # Alerts
    alerts:
      poor_performance_threshold: 0.5  # Alert if best Sharpe < 0.5
      time_limit_warning: 0.9          # Alert at 90% of time limit
```

### Optimization Diagnostics

```yaml
optimization:
  diagnostics:
    enabled: true
    
    # Parameter analysis
    parameter_analysis:
      sensitivity_analysis: true       # Which parameters matter most
      interaction_analysis: true       # Parameter interactions
      correlation_analysis: true       # Parameter correlations
      
    # Convergence analysis
    convergence_analysis:
      plot_convergence: true          # Objective vs trial number
      detect_plateaus: true           # When optimization stops improving
      early_stopping: true            # Stop if no improvement
      patience: 50                    # Trials without improvement
      
    # Validation
    validation:
      cross_validation: true          # K-fold validation of results
      bootstrap_confidence: true      # Confidence intervals
      out_of_sample_test: true        # Final validation
```

## 🎯 Multi-Stage Optimization

### Coarse-to-Fine Optimization

```yaml
# Stage 1: Coarse grid search
workflow:
  type: "multi_stage_optimization"
  
  stages:
    - name: "coarse_search"
      method: "grid"
      parameters:
        fast_period: [5, 15, 25]       # Coarse grid
        slow_period: [20, 40, 60]
        signal_threshold: [0.01, 0.05]
      objective: "sharpe_ratio"
      
    - name: "fine_search"
      method: "bayesian"
      n_trials: 100
      # Use best region from coarse search
      parameters:
        fast_period: ["${coarse_search.best_fast_period} ± 5"]
        slow_period: ["${coarse_search.best_slow_period} ± 10"]
        signal_threshold: ["${coarse_search.best_signal_threshold} ± 0.02"]
        
    - name: "final_validation"
      method: "walk_forward"
      parameters: "${fine_search.best_parameters}"
```

### Feature Selection + Optimization

```yaml
# Combined feature selection and parameter optimization
workflow:
  type: "feature_optimization"
  
  phases:
    - name: "feature_selection"
      type: "feature_selection"
      method: "recursive_elimination"
      max_features: 10
      
      feature_candidates:
        - "sma_5"
        - "sma_10" 
        - "rsi_14"
        - "macd"
        - "volume_sma"
        # ... more features
        
    - name: "parameter_optimization"
      type: "optimization"
      inputs: ["feature_selection.selected_features"]
      method: "bayesian"
      
      parameters:
        # Only optimize parameters for selected features
        dynamic_parameters: true
```

## 📈 Optimization Results Analysis

### Results Structure

```yaml
# Optimization results are automatically saved
results:
  best_parameters:
    fast_period: 12
    slow_period: 28
    signal_threshold: 0.025
    
  best_performance:
    sharpe_ratio: 1.85
    total_return: 0.234
    max_drawdown: -0.087
    
  optimization_history:
    - trial: 1
      parameters: {...}
      performance: {...}
    # ... all trials
    
  parameter_importance:
    signal_threshold: 0.45           # Most important
    fast_period: 0.32
    slow_period: 0.23
    
  convergence_analysis:
    converged: true
    convergence_trial: 78
    improvement_threshold: 0.01
```

### Visualization and Analysis

```yaml
reporting:
  optimization_analysis:
    # Parameter space visualization
    parameter_plots:
      - "parameter_heatmap"           # 2D parameter relationships
      - "parameter_scatter"           # Parameter vs objective
      - "parameter_importance"        # Feature importance plot
      
    # Performance analysis
    performance_plots:
      - "convergence_plot"            # Objective improvement over time
      - "distribution_plot"           # Distribution of results
      - "pareto_frontier"             # Multi-objective results
      
    # Validation plots
    validation_plots:
      - "out_of_sample_performance"
      - "parameter_stability"
      - "robustness_analysis"
```

## ⚠️ Optimization Pitfalls

### Overfitting Detection

```yaml
optimization:
  overfitting_protection:
    # Data splitting
    train_test_split: 0.8            # 80% train, 20% test
    cross_validation:
      enabled: true
      folds: 5
      
    # Performance degradation checks
    degradation_thresholds:
      sharpe_ratio_drop: 0.3         # Flag if test Sharpe drops 30%
      return_drop: 0.5               # Flag if test return drops 50%
      
    # Complexity penalties
    complexity_penalty:
      enabled: true
      penalty_per_parameter: 0.01    # Penalize complex models
      
    # Robustness testing
    robustness_tests:
      parameter_noise: 0.1           # Test ±10% parameter changes
      data_bootstrap: 100            # Bootstrap data variations
```

### Common Issues and Solutions

```yaml
optimization_issues:
  # Issue: Optimization gets stuck in local minima
  local_minima:
    solutions:
      - use_random_restarts: true
      - increase_exploration: true
      - try_genetic_algorithm: true
      
  # Issue: Results don't generalize
  poor_generalization:
    solutions:
      - increase_regularization: true
      - use_cross_validation: true
      - simplify_parameter_space: true
      - longer_out_of_sample_period: true
      
  # Issue: Optimization is too slow
  slow_optimization:
    solutions:
      - use_signal_replay: true
      - reduce_parameter_space: true
      - increase_parallel_workers: true
      - use_early_stopping: true
```

## 🎯 Optimization Best Practices

### 1. **Start Simple**
```yaml
# Begin with coarse grid, then refine
optimization_strategy:
  phase_1: "coarse_grid"             # Explore parameter space
  phase_2: "bayesian_refinement"     # Focus on promising regions
  phase_3: "robustness_testing"      # Validate results
```

### 2. **Use Appropriate Sample Sizes**
```yaml
data_requirements:
  minimum_trades: 100               # Statistical significance
  minimum_years: 3                  # Multiple market cycles
  out_of_sample_pct: 0.2           # 20% for final validation
```

### 3. **Consider Transaction Costs**
```yaml
# Always include realistic costs
execution:
  slippage_bps: 10
  commission_per_share: 0.005
  market_impact: true
```

### 4. **Validate Thoroughly**
```yaml
validation:
  methods:
    - "out_of_sample"
    - "walk_forward"
    - "monte_carlo"
    - "parameter_stability"
```

## 🤔 Common Questions

**Q: How many parameter combinations should I test?**
A: Start with 50-100 for grid search, 200-500 for Bayesian. More if using signal replay.

**Q: Should I optimize on the full dataset?**
A: No! Always reserve 20-30% for out-of-sample validation.

**Q: Which optimization method is best?**
A: Grid search for small spaces, Bayesian for larger continuous spaces, genetic algorithms for complex interactions.

**Q: How do I know if my optimization overfit?**
A: Compare in-sample vs out-of-sample performance. Large degradation indicates overfitting.

## 📝 Optimization Checklist

- [ ] Define clear optimization objective
- [ ] Reserve data for out-of-sample testing
- [ ] Include realistic transaction costs
- [ ] Use appropriate optimization method
- [ ] Set reasonable parameter ranges
- [ ] Monitor for overfitting
- [ ] Validate results thoroughly
- [ ] Document optimization process

## 📈 Next Steps

- **Validate Results**: [Walk-Forward Analysis](walk-forward-analysis.md)
- **Build Complex Workflows**: [Multi-Phase Workflows](multi-phase-workflows.md)
- **Deploy Optimized Strategy**: [Live Trading Guide](live-trading.md)

---

Continue to [Walk-Forward Analysis](walk-forward-analysis.md) to validate your optimized strategies →