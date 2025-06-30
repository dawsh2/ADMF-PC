"""Detailed analysis of best 5m swing pivot strategies"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_bfe07b48")
signal_dir = workspace / "traces/SPY_5m_1m/signals/swing_pivot_bounce_zones"

print("=== DETAILED ANALYSIS: BEST 5M SWING PIVOT STRATEGIES ===\n")

# Best strategies from our analysis
best_strategies = [
    (92, "Highest Edge (1.89 bps)"),
    (87, "Best Overall (Edge × Frequency)"),
    (88, "Balanced High Edge"),
    (20, "Most Active Profitable")
]

for strategy_id, description in best_strategies:
    print(f"\n{'='*60}")
    print(f"STRATEGY {strategy_id}: {description}")
    print('='*60)
    
    signal_file = signal_dir / f"SPY_5m_compiled_strategy_{strategy_id}.parquet"
    signals = pd.read_parquet(signal_file)
    
    # Detailed trade analysis
    trades = []
    current_position = None
    
    for j in range(len(signals)):
        curr = signals.iloc[j]
        
        if current_position is None and curr['val'] != 0:
            # Entry
            current_position = {
                'entry_idx': curr['idx'],
                'entry_price': curr['px'],
                'direction': curr['val'],
                'entry_row': j
            }
        elif current_position is not None and (curr['val'] == 0 or np.sign(curr['val']) != np.sign(current_position['direction'])):
            # Exit
            pct_return = (curr['px'] / current_position['entry_price'] - 1) * current_position['direction'] * 100
            bars_held = curr['idx'] - current_position['entry_idx']
            
            trades.append({
                'entry_idx': current_position['entry_idx'],
                'exit_idx': curr['idx'],
                'pct_return': pct_return,
                'bars_held': bars_held,
                'direction': 'long' if current_position['direction'] > 0 else 'short',
                'entry_price': current_position['entry_price'],
                'exit_price': curr['px']
            })
            
            # Reset position
            if curr['val'] != 0:
                current_position = {
                    'entry_idx': curr['idx'],
                    'entry_price': curr['px'],
                    'direction': curr['val'],
                    'entry_row': j
                }
            else:
                current_position = None
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        print(f"\nTrade Statistics:")
        print(f"- Total trades: {len(trades_df)}")
        print(f"- Avg return: {trades_df['pct_return'].mean():.4f}% ({trades_df['pct_return'].mean()*100:.2f} bps)")
        print(f"- Win rate: {(trades_df['pct_return'] > 0).mean():.1%}")
        print(f"- Profit factor: {trades_df[trades_df['pct_return'] > 0]['pct_return'].sum() / abs(trades_df[trades_df['pct_return'] < 0]['pct_return'].sum()):.2f}")
        
        # Direction breakdown
        print(f"\nDirection Breakdown:")
        for direction in ['long', 'short']:
            dir_trades = trades_df[trades_df['direction'] == direction]
            if len(dir_trades) > 0:
                print(f"- {direction.capitalize()}: {len(dir_trades)} trades, "
                      f"{dir_trades['pct_return'].mean()*100:.2f} bps avg, "
                      f"{(dir_trades['pct_return'] > 0).mean():.1%} win rate")
        
        # Holding period analysis
        print(f"\nHolding Period Analysis:")
        print(f"- Avg bars held: {trades_df['bars_held'].mean():.1f} (5-min bars)")
        print(f"- Avg time held: {trades_df['bars_held'].mean() * 5:.1f} minutes")
        print(f"- Shortest trade: {trades_df['bars_held'].min()} bars ({trades_df['bars_held'].min() * 5} minutes)")
        print(f"- Longest trade: {trades_df['bars_held'].max()} bars ({trades_df['bars_held'].max() * 5} minutes)")
        
        # Return distribution
        print(f"\nReturn Distribution:")
        print(f"- Best trade: {trades_df['pct_return'].max():.2f}%")
        print(f"- Worst trade: {trades_df['pct_return'].min():.2f}%")
        print(f"- Std dev: {trades_df['pct_return'].std():.4f}%")
        print(f"- Sharpe ratio: {trades_df['pct_return'].mean() / trades_df['pct_return'].std():.2f}")
        
        # Annual projections
        bars_per_year = 78 * 252  # 78 5-min bars per day
        trades_per_year = len(trades_df) * (bars_per_year / 16614)  # Scale to annual
        
        print(f"\nAnnual Projections:")
        print(f"- Trades per year: {trades_per_year:.0f}")
        print(f"- Trades per day: {trades_per_year/252:.1f}")
        
        # Calculate annual returns with costs
        avg_return_decimal = trades_df['pct_return'].mean() / 100
        
        print(f"\nExpected Annual Returns:")
        for cost_bps in [0, 0.5, 1, 2, 5]:
            net_return = avg_return_decimal - (cost_bps / 10000 * 2)  # Round trip
            if net_return > 0:
                annual_return = (1 + net_return) ** trades_per_year - 1
                print(f"- {cost_bps} bps cost: {annual_return:.1%}")
            else:
                print(f"- {cost_bps} bps cost: NEGATIVE")

# Summary comparison
print(f"\n\n{'='*60}")
print("SUMMARY: 5-MINUTE VS 1-MINUTE COMPARISON")
print('='*60)

print("\n1-Minute Results:")
print("- Baseline: 0.04 bps per trade")
print("- With high vol filter: 1.26 bps per trade")
print("- Trades: ~148 per month")

print("\n5-Minute Results:")
print("- Best edge: 1.89 bps per trade (50% improvement)")
print("- Best overall: 1.03 bps with 138 trades")
print("- More parameter combinations found profitable")

print("\nKey Improvements on 5-minute:")
print("1. Higher edge per trade (1.89 vs 1.26 bps)")
print("2. Better win rates (65.7% vs 47.3%)")
print("3. More robust across parameters")
print("4. Cleaner support/resistance levels")
print("5. Lower execution costs (fewer trades)")

print("\nRecommendation:")
print("The 5-minute swing pivot strategy shows significant improvement")
print("over 1-minute, with edges approaching 2 bps and win rates > 65%.")
print("Combined with your 15-19 bps Bollinger strategy, this creates")
print("a strong two-strategy portfolio on 5-minute timeframe.")