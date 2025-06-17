# Sparse Trace Analysis Module

## Overview

This module provides analytics tools for analyzing sparse trace data from ADMF-PC backtests. The data follows a specific sparse storage protocol optimized for efficiency.

## Data Format Understanding

### Signal Storage (Sparse)
Signals are stored as **changes only**, not every bar:
- **First signal**: Non-zero value represents opening a position (includes price)
- **Subsequent signals**: Represent position changes (close previous, optionally open new)
- **Signal transitions**:
  - `1 @ $520` → `0 @ $530` = Close long position (+$10 profit)
  - `1 @ $520` → `-1 @ $530` = Close long (+$10) AND open short simultaneously
  - `0 @ $520` → `1 @ $530` = Open long position

### Classifier Storage (Sparse)
Classifiers broadcast **state changes only**:
- `BULLISH at bar 100` (initial state)
- `BEARISH at bar 150` = BULLISH lasted 50 bars (150-100)
- `SIDEWAYS at bar 175` = BEARISH lasted 25 bars (175-150)

### File Structure
```
traces/
├── SPY_1m/
│   ├── signals/
│   │   ├── strategy_type/
│   │   │   └── SPY_strategy_name_params.parquet
│   └── classifiers/
│       ├── classifier_type/
│       │   └── SPY_classifier_name_params.parquet
```

### Parquet Schema
**Signal Files:**
- `idx`: Bar index (int)
- `val`: Signal value (-1, 0, 1 or continuous)
- `px`: Asset price at signal time (float)
- `ts`: Timestamp
- `sym`: Symbol
- `strat`: Strategy identifier
- Additional metadata fields

**Classifier Files:**
- `idx`: Bar index where state change occurred
- `val`: New classifier state (string)
- `px`: Asset price at state change
- `ts`: Timestamp
- Additional metadata fields

## Performance Calculation Methods

### Log Returns Per Trade (Recommended)
```python
# For each trade
trade_log_return = log(exit_price / entry_price) * signal_value

# Apply execution costs (multiplicative)
if execution_cost_multiplier:
    trade_log_return *= execution_cost_multiplier  # e.g., 0.97 for 3% cost

# Apply execution costs (additive)
if execution_cost_additive:
    linear_return = exp(trade_log_return) - 1
    linear_return -= execution_cost_additive / entry_price
    trade_log_return = log(1 + linear_return)

# Sum all trades
total_log_return = sum(all_trade_log_returns)
percentage_return = exp(total_log_return) - 1
```

### Execution Cost Handling
**Multiplicative Costs** (preferred for percentage-based costs):
```python
cost_multiplier = 0.97  # 3% total cost per trade
net_return = gross_return * cost_multiplier
```

**Additive Costs** (for fixed dollar amounts):
```python
cost_per_trade = 2.0  # $2 per trade
net_pnl = gross_pnl - cost_per_trade
```

## Regime Attribution Rules

### Trade Attribution
- **Attribute trades to the regime where position was OPENED**
- Use sparse classifier changes to determine active regime at opening bar
- Ignore regime changes during the trade duration

### Implementation
```python
def get_regime_at_bar(bar_idx, classifier_changes):
    # Find most recent state change before or at bar_idx
    relevant = classifier_changes[classifier_changes['bar_idx'] <= bar_idx]
    return relevant.iloc[-1]['state'] if len(relevant) > 0 else 'unknown'
```

## Classifier Balance Analysis

### Duration Calculation
Calculate actual time spent in each state:
```python
for i in range(len(state_changes)):
    current_bar = state_changes.iloc[i]['bar_idx']
    current_state = state_changes.iloc[i]['state']
    
    if i < len(state_changes) - 1:
        next_bar = state_changes.iloc[i + 1]['bar_idx']
        duration = next_bar - current_bar
    else:
        duration = estimated_end_bar - current_bar
    
    state_durations[current_state] += duration
```

### Balance Metrics
- **Balance Score**: Sum of absolute deviations from ideal percentage (lower = better)
- **Normalized Entropy**: Entropy / log(num_states) (higher = better, 0-1 scale)
- **Min/Max Percentage**: Identifies severely imbalanced states

### Selection Criteria
- Balance Score < 50 (reasonable balance)
- Min state percentage > 10% (avoid rarely-triggered states)
- Normalized entropy > 0.7 (good distribution)

## Common Analysis Patterns

### 1. Classifier Evaluation
```python
from sparse_trace_analysis import ClassifierAnalyzer

analyzer = ClassifierAnalyzer(workspace_path)
analysis = analyzer.analyze_all_classifiers()
balanced_classifiers = analyzer.select_balanced_classifiers(analysis)
```

### 2. Strategy Performance by Regime
```python
from sparse_trace_analysis import StrategyAnalyzer

analyzer = StrategyAnalyzer(workspace_path)
results = analyzer.analyze_regime_performance(
    classifier_name="best_classifier",
    strategy_files=strategy_list,
    execution_cost_multiplier=0.99  # 1% cost
)
```

### 3. Regime Transition Analysis
```python
from sparse_trace_analysis import RegimeAnalyzer

analyzer = RegimeAnalyzer(workspace_path)
transitions = analyzer.analyze_regime_transitions(classifier_name)
persistence = analyzer.calculate_regime_persistence(transitions)
```

## Best Practices

### Data Validation
- Always verify bar_idx ordering (should be ascending)
- Check for missing price data
- Validate signal values are within expected ranges
- Ensure classifier states are consistent

### Performance Considerations
- Use pandas merge_asof for regime attribution (efficient for sparse data)
- Cache classifier state lookups for repeated analysis
- Process strategies in batches to manage memory
- Consider parallel processing for large strategy sets

### Error Handling
- Handle edge cases: empty data, missing classifiers, invalid signals
- Graceful degradation when regime data is incomplete
- Logging for debugging sparse data issues

## Module Components

- `classifier_analysis.py`: Classifier balance and duration analysis
- `strategy_analysis.py`: Strategy performance calculation and regime attribution  
- `performance_calculation.py`: Log returns and execution cost handling
- `regime_attribution.py`: Regime mapping and transition analysis
- `data_validation.py`: Input validation and error checking

## Usage Examples

See individual module files for detailed usage examples and API documentation.