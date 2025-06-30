# Trade-Based vs Bar-Based Metrics: Critical Fix

## The Problem

We discovered a fundamental inconsistency in how performance metrics were calculated:

1. **Base performance metrics**: Used bar-by-bar calculations
   - Win rate: ~27% (percentage of profitable bars)
   - Returns: Calculated on every bar with position

2. **Stop loss analysis**: Used trade-by-trade calculations
   - Win rate: 60-70% (percentage of profitable trades)
   - Returns: Calculated on completed trades

This made comparisons meaningless - we were comparing apples to oranges.

## The Root Cause

### Bar-Based Calculation (OLD)
```python
# Counts every bar where strategy had position
df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
win_rate = (df['strategy_returns'] > 0).sum() / len(df[df['strategy_returns'] != 0])
```

Example: A 10-bar winning trade shows:
- 10 individual bar returns (some positive, some negative)
- If 3 bars were negative: 30% bar-based "loss rate"
- Overall trade was profitable

### Trade-Based Calculation (NEW)
```python
# Counts actual completed trades
trades = extract_trades(strategy_hash, trace_path, market_data)
win_rate = len(trades[trades['net_return'] > 0]) / len(trades)
```

Example: Same 10-bar trade shows:
- 1 completed trade with net profit
- 100% trade-based win rate

## The Fix

### 1. Updated Performance Calculation
Modified `calculate_performance()` in `universal_analysis.ipynb` to use trade-based metrics consistently.

### 2. Created Analysis Tools
- `corrected_performance_analysis.py`: Recalculates all metrics using trades
- `trade_vs_bar_metrics_comparison.py`: Shows the difference between methods

### 3. Key Changes
- Win rate: Now counts winning trades / total trades
- Profit factor: Sum of winning trades / |sum of losing trades|
- Sharpe ratio: Calculated from trade returns with proper annualization
- All metrics now consistent with stop loss analysis

## Impact on Results

### Before Fix
- Base win rate: ~27%
- Stop loss win rate: 60-70%
- Confusion: "Why does adding stop loss triple the win rate?"

### After Fix
- Base win rate: 60-70% (trade-based)
- Stop loss win rate: 60-70% (trade-based)
- Clear comparison: Stop losses affect trade outcomes, not calculation method

## Usage

### In Jupyter Notebook
```python
# Force recalculation with new method
IGNORE_CACHE = True

# Or delete old cache
!rm -f performance_metrics.parquet

# Load corrected analysis
%load /Users/daws/ADMF-PC/src/analytics/snippets/corrected_performance_analysis.py
```

### Understanding Your Results
- Win rates of 60-70% are normal for trend-following strategies (by trade count)
- Bar-based win rates of 20-30% are also normal (many bars go against position)
- Always use trade-based metrics for:
  - Strategy comparison
  - Stop loss optimization
  - Portfolio analysis

## Profit Factor Calculation

### Correct Formula
```
Profit Factor = Sum of all winning trade returns / |Sum of all losing trade returns|
```

### Example
- 3 winning trades: +2%, +3%, +1% = +6% total
- 2 losing trades: -1%, -0.5% = -1.5% total
- Profit Factor = 6% / 1.5% = 4.0

A profit factor > 1.0 means the strategy makes more on winners than it loses on losers.

## Recommendations

1. **Always use trade-based metrics** for performance evaluation
2. **Delete old performance caches** when switching calculation methods
3. **Verify win rates** are in reasonable ranges (50-70% for most strategies)
4. **Check profit factors** are calculated from trades, not bars

## Next Steps

1. Re-run any backtests with the corrected calculation
2. Re-evaluate stop loss optimization with consistent metrics
3. Focus on the refined Bollinger parameters (std_dev ~1.0)