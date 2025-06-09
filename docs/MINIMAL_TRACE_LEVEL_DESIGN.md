# Minimal Trace Level Design

## Overview

The `minimal` trace level is designed for memory-efficient backtesting, particularly for large-scale parameter optimization. It tracks the absolute minimum data needed to calculate performance metrics.

## Core Design

### Only Portfolio Containers Trace
- **Portfolio containers**: Event tracing enabled with `retention_policy='trade_complete'`
- **All other containers**: Tracing disabled (data, feature, strategy, risk, execution)

### Trade Lifecycle Tracking
Events are kept in memory only while a trade is open:
1. **Trade Opens**: Events stored in `active_trades[trade_id]`
2. **Trade Closes**: 
   - Metrics updated (win/loss, P&L, etc.)
   - All events for that trade deleted
3. **Result**: Only open trades consume memory

### No Event Limits
- `max_events: 0` - No artificial limits
- Memory naturally bounded by number of open positions
- Retention policy handles cleanup automatically

## Persistent Metrics (Always in Memory)

The `StreamingMetrics` class maintains these metrics incrementally:

### What We Track
- **Trade Statistics**: 
  - Total trades (`n_trades`)
  - Winning trades (`winning_trades`) 
  - Losing trades (`losing_trades`)
  - Win rate (calculated from above)
- **P&L Metrics**:
  - Total P&L (`total_pnl`)
  - Gross profit (`gross_profit`)
  - Gross loss (`gross_loss`)
  - Profit factor (calculated)
- **Return Statistics**:
  - Total return
  - Mean return (Welford's algorithm)
  - Variance/Sharpe (Welford's algorithm)
- **Portfolio State**:
  - Current value
  - Initial capital

### What We DON'T Track (Tradeoffs)
- **Max Drawdown**: Requires full equity curve
- **Equity Curve**: No historical snapshots
- **Trade History**: Individual trades not stored
- **Detailed Event History**: No replay capability

## Memory Usage Pattern

```
# Start of backtest
Memory: [StreamingMetrics only] ~1KB

# Open position in SPY
Memory: [StreamingMetrics] + [SPY order/fill events] ~10KB

# Open position in QQQ  
Memory: [StreamingMetrics] + [SPY events] + [QQQ events] ~20KB

# Close SPY position
Memory: [StreamingMetrics (updated)] + [QQQ events] ~10KB
        ↑ SPY events deleted after updating win/loss, P&L

# Close QQQ position
Memory: [StreamingMetrics (updated)] ~1KB
        ↑ All trade events deleted
```

## Implementation Details

### MetricsEventTracer Configuration
```python
{
    'retention_policy': 'trade_complete',
    'max_events': 0,  # No limit
    'store_trades': False,
    'store_equity_curve': False
}
```

### Event Flow
1. Portfolio receives FILL event
2. If opening: Store in `active_trades[correlation_id]`
3. If closing: 
   - Call `update_from_trade()` to update metrics
   - Call `_cleanup_completed_trade_events()`
   - Delete from `active_trades`

### Correlation ID Usage
- Events use `correlation_id` to group related events
- Falls back to extracting `order_id` from payload
- All events for a trade share the same correlation

## Use Cases

### Perfect For
- Large parameter sweeps (1000s of portfolios)
- Walk-forward optimization
- Memory-constrained environments
- Production systems needing only final metrics

### Not Suitable For
- Debugging execution issues
- Analyzing individual trades
- Calculating max drawdown
- Equity curve visualization

## Future Enhancement: POSITION_OPENED/CLOSED Events

We added `POSITION_OPENED` and `POSITION_CLOSED` event types to distinguish:
- **POSITION_OPENED**: New position established
- **POSITION_CLOSED**: Position fully closed
- **PORTFOLIO_UPDATE**: Value changes from price movements

This makes it easier to:
1. Track exactly when to clean up trade events
2. Distinguish trade events from price updates
3. Potentially track position count without full history

## Summary

The minimal trace level achieves:
- **Minimal Memory**: Only open trades in memory
- **Full Metrics**: Everything except max drawdown
- **No Limits**: Natural memory management
- **Simple Config**: Just `trace_level: minimal`

This is the ideal setting for large-scale optimization where memory efficiency is critical and you only need final performance metrics.