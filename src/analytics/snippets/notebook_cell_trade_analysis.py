# Notebook cell for detailed trade-by-trade analysis
# Copy and paste this into a new cell in your analysis notebook

# Import the analysis functions
import sys
sys.path.append('/Users/daws/ADMF-PC')
from src.analytics.snippets.trade_by_trade_analysis import analyze_trades_detailed, display_trade_comparison

# Analyze trades for strategy 5edc4365
strategy_hash = '5edc43651004'
strategy_info = strategy_index[strategy_index['strategy_hash'] == strategy_hash].iloc[0]
trace_path = strategy_info['trace_path']

print(f"Analyzing strategy: {strategy_hash[:8]}")
print(f"Parameters: period={strategy_info['period']}, std_dev={strategy_info['std_dev']}")
print(f"Trace path: {trace_path}")
print("\n")

# Analyze first 100 trades with detailed information
trades = analyze_trades_detailed(
    strategy_hash=strategy_hash,
    trace_path=trace_path,
    market_data=market_data,
    stop_pct=0.075,
    target_pct=0.1,
    execution_cost_bps=1.0,
    max_trades=100,
    run_dir=run_dir
)

# Display formatted comparison
display_trade_comparison(trades, num_trades=30)

# Show some specific examples where stops/targets made a difference
print("\n" + "="*120)
print("DETAILED EXAMPLES OF STOP/TARGET EXITS")
print("-"*120)

# Find trades that hit stops
stop_trades = trades[trades['exit_type'] == 'stop_loss'].head(3)
print("\nExample STOP LOSS trades:")
for _, trade in stop_trades.iterrows():
    print(f"\nTrade #{trade['trade_num']}:")
    print(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f} ({trade['direction']})")
    print(f"  Stop price: ${trade['stop_price']:.2f}")
    print(f"  Exit bar #{trade['bars_in_trade']}: {trade['actual_exit_time']}")
    print(f"  Original exit would have been: bar #{trade['total_bars_without_stops']} @ ${trade['original_exit_price']:.2f}")
    print(f"  Saved loss: {(trade['original_net_return'] - trade['actual_net_return'])*100:.3f}%")

# Find trades that hit targets
target_trades = trades[trades['exit_type'] == 'take_profit'].head(3)
print("\nExample TAKE PROFIT trades:")
for _, trade in target_trades.iterrows():
    print(f"\nTrade #{trade['trade_num']}:")
    print(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f} ({trade['direction']})")
    print(f"  Target price: ${trade['target_price']:.2f}")
    print(f"  Exit bar #{trade['bars_in_trade']}: {trade['actual_exit_time']}")
    print(f"  Original exit would have been: bar #{trade['total_bars_without_stops']} @ ${trade['original_exit_price']:.2f}")
    print(f"  Captured profit: {(trade['actual_net_return'] - trade['original_net_return'])*100:.3f}%")

# Save to CSV for comparison with execution engine
output_file = f'trades_analysis_{strategy_hash[:8]}.csv'
trades.to_csv(output_file, index=False)
print(f"\n✅ Saved detailed trade analysis to: {output_file}")

# Create a summary for easy comparison
summary = {
    'total_trades': len(trades),
    'stop_exits': len(trades[trades['exit_type'] == 'stop_loss']),
    'target_exits': len(trades[trades['exit_type'] == 'take_profit']),
    'signal_exits': len(trades[trades['exit_type'] == 'signal']),
    'avg_return_no_stops': trades['original_net_return'].mean() * 100,
    'avg_return_with_stops': trades['actual_net_return'].mean() * 100,
    'win_rate_no_stops': (trades['original_net_return'] > 0).mean() * 100,
    'win_rate_with_stops': (trades['actual_net_return'] > 0).mean() * 100,
    'total_return_no_stops': ((1 + trades['original_net_return']).prod() - 1) * 100,
    'total_return_with_stops': ((1 + trades['actual_net_return']).prod() - 1) * 100
}

print("\n" + "="*60)
print("SUMMARY FOR COMPARISON WITH EXECUTION ENGINE")
print("-"*60)
for key, value in summary.items():
    if 'return' in key or 'rate' in key:
        print(f"{key:>25}: {value:>8.2f}%")
    else:
        print(f"{key:>25}: {value:>8}")