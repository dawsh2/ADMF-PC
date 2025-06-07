"""
Demonstrate the MINIMAL trace level for memory-efficient backtesting.

Shows how portfolio containers track only open trades, update metrics
when trades close, then discard event traces to minimize memory usage.
"""

# Example configuration using minimal trace level
minimal_config = """
workflow: simple_backtest
trace_level: minimal  # Only track open trades

data:
  symbols: [SPY, QQQ, IWM]
  start_date: '2023-01-01'
  end_date: '2023-12-31'
  
strategies:
  - type: momentum
    parameters:
      fast_period: 10
      slow_period: 20

portfolios:
  - name: main_portfolio
    initial_capital: 100000
    
execution:
  slippage_bps: 5
  commission_per_share: 0.01
"""

# What happens with trace_level: minimal

print("""
=== MINIMAL Trace Level Behavior ===

With trace_level: minimal, the system optimizes for memory efficiency:

1. Portfolio Containers:
   - Only track events for OPEN trades
   - When a trade closes:
     * Update metrics (Sharpe, returns, drawdown, etc.)
     * Remove all events for that trade from memory
   - Metrics persist in memory (lightweight)
   - No trade history stored
   - No equity curve snapshots

2. Other Containers:
   - Data containers: NO tracing (disabled)
   - Feature containers: NO tracing (disabled)
   - Strategy containers: NO tracing (stateless anyway)
   - Risk containers: NO tracing (stateless anyway)
   - Execution container: NO tracing (disabled)

3. Memory Usage Pattern:
   
   Start of backtest:
   └── Memory: [Metrics only]
   
   Open position in SPY:
   └── Memory: [Metrics] + [SPY trade events]
   
   Open position in QQQ:
   └── Memory: [Metrics] + [SPY trade events] + [QQQ trade events]
   
   Close SPY position:
   └── Memory: [Metrics (updated)] + [QQQ trade events]
       * SPY events deleted after metrics update
   
   Close QQQ position:
   └── Memory: [Metrics (updated)]
       * All trade events deleted
       
4. Perfect for:
   - Large-scale parameter optimization
   - Walk-forward analysis with many windows
   - Multi-asset backtests with limited memory
   - Production systems that only need final metrics

5. What You Get:
   - Full performance metrics (Sharpe, returns, etc.)
   - Minimal memory footprint
   - Fast execution
   - No post-processing needed for metrics

6. What You DON'T Get:
   - Individual trade history
   - Equity curve visualization
   - Event replay capability
   - Detailed debugging information

=== Implementation Details ===

The MetricsEventTracer with retention_policy='trade_complete':

1. Stores events by correlation_id (or order_id)
2. When a FILL closes a position:
   - Calculates P&L for the trade
   - Updates streaming metrics
   - Calls _cleanup_completed_trade_events()
   - Removes all events for that trade

3. StreamingMetrics maintains:
   - Total return, Sharpe ratio
   - Win rate, profit factor
   - Maximum drawdown
   - All calculated incrementally (no history needed)

This achieves the absolute minimum memory usage while still
providing complete performance metrics.
""")

# Show the actual configuration applied
print("\n=== Actual Configuration Applied ===\n")
print("Portfolio containers get:")
print("""
{
    "enabled": True,
    "max_events": 0,  # No limit - retention policy manages memory
    "retention_policy": "trade_complete",
    "results": {
        "streaming_metrics": True,
        "retention_policy": "trade_complete",
        "store_trades": False,      # No trade history
        "store_equity_curve": False  # No equity snapshots
    }
}
""")

print("\nAll other containers:")
print("- Data/Feature/Strategy/Risk/Execution: Tracing DISABLED")
print("- Only portfolio containers track events in MINIMAL mode")

print("\n=== Memory Savings Example ===")
print("""
Traditional approach (store everything):
- 1 year backtest, 1000 trades
- ~10 events per trade = 10,000 events
- Each event ~1KB = 10MB per portfolio
- 100 portfolios = 1GB memory

Minimal trace level:
- Same backtest
- Max 10 open positions at once
- ~100 events in memory at peak
- Metrics: ~1KB per portfolio  
- 100 portfolios = 100KB memory (10,000x reduction!)
""")

# Compare with other trace levels
print("\n=== Trace Level Comparison ===\n")
print("NONE:    No tracing, no metrics (except what containers track)")
print("MINIMAL: Metrics only, discard events after trades close")  
print("NORMAL:  Metrics + trade history + equity curve")
print("DEBUG:   Everything stored for analysis")

print("\n=== Usage Recommendations ===")
print("""
Use MINIMAL when:
- Running large parameter sweeps
- Memory is limited
- You only need final performance metrics
- Running in production

Use NORMAL when:
- Need to analyze individual trades
- Want equity curve visualization
- Developing new strategies
- Running smaller backtests

Use DEBUG when:
- Debugging issues
- Need complete event history
- Analyzing execution quality
- One-off analysis runs
""")