# Walk-Forward Validation in ADMF-PC

## Overview

Walk-forward validation is a robust method for testing trading strategies that simulates real-world trading conditions by training on historical data and testing on future unseen data. This approach helps detect overfitting and validates strategy robustness over time.

## Key Concepts

### Rolling vs Anchored Walk-Forward

#### Rolling Walk-Forward (Fixed Window)
- Training window maintains constant size
- Both training and test windows roll forward
- Suitable for strategies that adapt to changing market conditions

```
Period 0: train[0:500], test[500:600]
Period 1: train[100:600], test[600:700]  # Window rolled forward
Period 2: train[200:700], test[700:800]
```

#### Anchored Walk-Forward (Expanding Window)
- Training always starts from the beginning
- Training window expands over time
- Test window rolls forward
- Suitable for strategies that benefit from more historical data

```
Period 0: train[0:500], test[500:600]
Period 1: train[0:600], test[600:700]  # Training expanded
Period 2: train[0:700], test[700:800]
```

## Implementation

### Basic Setup

```python
from src.strategy.optimization import (
    WalkForwardValidator,
    WalkForwardAnalyzer,
    GridOptimizer,
    SharpeObjective
)

# Create validator
validator = WalkForwardValidator(
    data_length=1000,     # Total data points
    train_size=600,       # Training window size
    test_size=100,        # Test window size
    step_size=100,        # Step between periods
    anchored=False        # Rolling (True for anchored)
)

# Create analyzer
analyzer = WalkForwardAnalyzer(
    validator=validator,
    optimizer=GridOptimizer(),
    objective=SharpeObjective(),
    backtest_func=your_backtest_function
)
```

### Running Analysis

```python
# Define strategy and parameters
results = analyzer.analyze_strategy(
    strategy_class='MomentumStrategy',
    base_params={'signal_cooldown': 3600},
    parameter_space={
        'lookback_period': [10, 20, 30, 40],
        'momentum_threshold': [0.01, 0.02, 0.03]
    },
    market_data=your_market_data
)

# Check results
print(f"Average train score: {results['aggregated']['train']['mean']:.3f}")
print(f"Average test score: {results['aggregated']['test']['mean']:.3f}")
print(f"Overfitting ratio: {results['aggregated']['overfitting_ratio']:.2f}")
print(f"Strategy robust: {results['summary']['robust']}")
```

## Integration with ADMF-PC Architecture

### Container Isolation

Each walk-forward period runs in complete isolation:

```python
# Containerized walk-forward
containerized = ContainerizedWalkForward(
    analyzer=analyzer,
    container_factory=BacktestContainerFactory
)

# Each period gets its own container
# - walkforward_period_0_train
# - walkforward_period_0_test
# - walkforward_period_1_train
# - walkforward_period_1_test
```

### Benefits of Container Isolation

1. **No State Leakage**: Each period is completely independent
2. **Parallel Execution**: Run multiple periods simultaneously
3. **Resource Management**: Each container has defined limits
4. **Reproducibility**: Identical container structure ensures consistent results

## Practical Examples

### Example 1: Quarterly Strategy Rebalancing

```python
# 2 years of data, train on 1 year, test on 3 months, rebalance quarterly
validator = WalkForwardValidator(
    data_length=504,      # 2 years of trading days
    train_size=252,       # 1 year training
    test_size=63,         # 3 months test
    step_size=63,         # Quarterly steps
    anchored=False
)
```

This creates 4 periods:
- Q1: Train on Year 1, Test on Q1 of Year 2
- Q2: Train on Q2 Year 1 - Q1 Year 2, Test on Q2 of Year 2
- Q3: Train on Q3 Year 1 - Q2 Year 2, Test on Q3 of Year 2
- Q4: Train on Q4 Year 1 - Q3 Year 2, Test on Q4 of Year 2

### Example 2: Monthly Parameter Updates

```python
# More frequent updates for volatile markets
validator = WalkForwardValidator(
    data_length=504,      # 2 years
    train_size=252,       # 1 year training
    test_size=21,         # 1 month test
    step_size=21,         # Monthly steps
    anchored=False
)
```

### Example 3: Expanding Window for Long-Term Strategies

```python
# Use all available history
validator = WalkForwardValidator(
    data_length=1260,     # 5 years
    train_size=252,       # Start with 1 year
    test_size=63,         # 3 months test
    step_size=63,         # Quarterly steps
    anchored=True         # Expanding window
)
```

## Interpreting Results

### Key Metrics

1. **Overfitting Ratio**: Train performance / Test performance
   - < 1.2: Excellent (minimal overfitting)
   - 1.2-1.5: Good (acceptable overfitting)
   - 1.5-2.0: Caution (significant overfitting)
   - > 2.0: Poor (severe overfitting)

2. **Consistency**: 1 - (Test StdDev / Test Mean)
   - > 0.7: Highly consistent
   - 0.5-0.7: Moderately consistent
   - < 0.5: Inconsistent

3. **Robustness**: Strategy is considered robust if:
   - Overfitting ratio < 1.5
   - Average test score > 0
   - Consistency > 0.5

### Example Output

```
Period 0:
  Optimal Parameters:
    - lookback_period: 20
    - momentum_threshold: 0.02
  Training Performance: 1.850
  Test Performance: 1.523
  Test Sharpe Ratio: 1.523
  Test Total Return: 12.3%
  Test Max Drawdown: 8.5%

Aggregate Results:
  Average Training Score: 1.920 ± 0.150
  Average Test Score: 1.480 ± 0.120
  Overfitting Ratio: 1.30

Summary:
  Number of Periods: 4
  Average Train Score: 1.920
  Average Test Score: 1.480
  Consistency: 91.9%
  Strategy Robust: YES
```

## Best Practices

### 1. Choose Appropriate Window Sizes

- **Training Size**: At least 1 year for daily data
- **Test Size**: 1-3 months (not too small or large)
- **Step Size**: Match your rebalancing frequency

### 2. Consider Market Regimes

Different market conditions may require different parameters:
- Bull markets: Shorter lookback periods
- Bear markets: Longer lookback periods
- High volatility: More conservative thresholds

### 3. Multiple Objectives

Use composite objectives for balanced optimization:

```python
objective = CompositeObjective([
    (SharpeObjective(), 0.6),      # 60% weight on Sharpe
    (CalmarObjective(), 0.3),      # 30% weight on Calmar
    (MinDrawdownObjective(), 0.1)  # 10% weight on drawdown
])
```

### 4. Parameter Stability

Check if optimal parameters are stable across periods:
- Large parameter swings indicate overfitting
- Stable parameters suggest robust strategy

### 5. Out-of-Sample Reserve

Keep final 10-20% of data as ultimate out-of-sample test:
- Never use this data during development
- Final validation only

## Integration with Optimization Workflow

Walk-forward validation integrates seamlessly with the multi-phase optimization workflow:

### Phase 1: Parameter Optimization
- Use walk-forward to find robust parameters
- Test across different market conditions

### Phase 2: Regime Analysis
- Compare parameters across regimes
- Identify regime-specific optimizations

### Phase 3: Weight Optimization
- Use walk-forward to validate ensemble weights
- Ensure weights are stable over time

### Phase 4: Final Validation
- Ultimate walk-forward on reserved data
- No further optimization allowed

## Performance Considerations

### Computational Efficiency

Walk-forward can be computationally intensive:
- N periods × M parameter combinations × Backtest time

Optimizations:
1. **Parallel Execution**: Run periods in parallel containers
2. **Signal Caching**: Save signals from Phase 1 for reuse
3. **Coarse-to-Fine**: Start with large steps, refine later
4. **Early Stopping**: Skip poor performing parameters

### Memory Management

With result streaming:
- Each period streams results to disk
- Only top performers kept in memory
- Aggregation done incrementally

## Conclusion

Walk-forward validation is essential for developing robust trading strategies. By combining it with ADMF-PC's container architecture, we achieve:

- **Realistic Testing**: Simulates actual trading conditions
- **Overfitting Detection**: Identifies strategies that won't work live
- **Parameter Stability**: Finds parameters that work across time
- **Clean Isolation**: No data leakage between periods
- **Scalable Execution**: Parallel processing of periods

Remember: A strategy that performs well in walk-forward validation has a much higher chance of success in live trading than one optimized on the entire dataset at once.