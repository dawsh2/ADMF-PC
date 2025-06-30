"""Analyze Keltner Bands with stop loss distribution from 0.01% to 0.5%"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_d5807cc2")
signal_file = workspace / "traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_0.parquet"

print("=== KELTNER BANDS STOP LOSS DISTRIBUTION ANALYSIS ===\n")

# Load signals
signals = pd.read_parquet(signal_file)

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})
spy_subset = spy_1m.iloc[:81787].copy()

def analyze_with_stop(stop_pct):
    """Analyze performance with a specific stop loss percentage"""
    trades = []
    entry_data = None
    
    for i in range(len(signals)):
        curr = signals.iloc[i]
        
        if entry_data is None and curr['val'] != 0:
            entry_data = {
                'idx': curr['idx'],
                'price': curr['px'],
                'signal': curr['val']
            }
        
        elif entry_data is not None:
            # Check each bar for stop loss
            start_idx = entry_data['idx'] + 1
            end_idx = min(curr['idx'] + 1, len(spy_subset))
            
            stopped_out = False
            exit_idx = curr['idx']
            exit_price = curr['px']
            
            for check_idx in range(start_idx, end_idx):
                if check_idx >= len(spy_subset):
                    break
                    
                check_bar = spy_subset.iloc[check_idx]
                
                # Check stop
                if entry_data['signal'] > 0:  # Long
                    stop_level = entry_data['price'] * (1 - stop_pct)
                    if check_bar['low'] <= stop_level:
                        stopped_out = True
                        exit_price = stop_level
                        exit_idx = check_idx
                        break
                else:  # Short
                    stop_level = entry_data['price'] * (1 + stop_pct)
                    if check_bar['high'] >= stop_level:
                        stopped_out = True
                        exit_price = stop_level
                        exit_idx = check_idx
                        break
            
            # Process exit
            should_exit = stopped_out or curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])
            
            if should_exit and entry_data['idx'] < len(spy_subset):
                pct_return = (exit_price / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = exit_idx - entry_data['idx']
                
                # For quick exits analysis
                is_quick = duration < 5
                
                trade = {
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'duration': duration,
                    'stopped_out': stopped_out,
                    'is_quick': is_quick
                }
                trades.append(trade)
                
                # Check next signal
                if curr['val'] != 0 and not stopped_out:
                    entry_data = {
                        'idx': curr['idx'],
                        'price': curr['px'],
                        'signal': curr['val']
                    }
                else:
                    entry_data = None
    
    return pd.DataFrame(trades)

# Test stop levels from 0.01% to 0.5%
stop_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
total_days = 81787 / 390

print("Analysis for ALL trades:\n")
print(f"{'Stop %':<8} {'Edge (bps)':<12} {'Win Rate':<10} {'Trades/Day':<12} {'Stop Rate':<12} {'Avg Duration':<12}")
print("-" * 76)

all_results = []

for stop_pct in stop_levels:
    trades_df = analyze_with_stop(stop_pct / 100)  # Convert percentage to decimal
    
    if len(trades_df) > 0:
        avg_return = trades_df['pct_return'].mean()
        win_rate = (trades_df['pct_return'] > 0).mean()
        tpd = len(trades_df) / total_days
        stop_rate = trades_df['stopped_out'].mean()
        avg_duration = trades_df['duration'].mean()
        
        all_results.append({
            'stop_pct': stop_pct,
            'edge_bps': avg_return * 100,  # Convert to actual basis points
            'win_rate': win_rate,
            'trades_per_day': tpd,
            'stop_rate': stop_rate,
            'avg_duration': avg_duration,
            'total_trades': len(trades_df)
        })
        
        print(f"{stop_pct:>6.2f}% {avg_return * 100:>11.2f} {win_rate:>9.1%} "
              f"{tpd:>11.1f} {stop_rate:>11.1%} {avg_duration:>11.1f}")

# Now analyze QUICK EXITS only (<5 bars)
print("\n\nAnalysis for QUICK EXITS (<5 bars) only:\n")
print(f"{'Stop %':<8} {'Edge (bps)':<12} {'Win Rate':<10} {'Trades/Day':<12} {'Stop Rate':<12} {'% of Total':<12}")
print("-" * 76)

quick_results = []

for stop_pct in stop_levels:
    trades_df = analyze_with_stop(stop_pct / 100)
    quick_trades = trades_df[trades_df['is_quick']]
    
    if len(quick_trades) > 0:
        avg_return = quick_trades['pct_return'].mean()
        win_rate = (quick_trades['pct_return'] > 0).mean()
        tpd = len(quick_trades) / total_days
        stop_rate = quick_trades['stopped_out'].mean()
        pct_of_total = len(quick_trades) / len(trades_df) * 100
        
        quick_results.append({
            'stop_pct': stop_pct,
            'edge_bps': avg_return * 100,
            'win_rate': win_rate,
            'trades_per_day': tpd,
            'stop_rate': stop_rate,
            'pct_of_total': pct_of_total
        })
        
        print(f"{stop_pct:>6.2f}% {avg_return * 100:>11.2f} {win_rate:>9.1%} "
              f"{tpd:>11.1f} {stop_rate:>11.1%} {pct_of_total:>10.1f}%")

# Find optimal stop levels
all_df = pd.DataFrame(all_results)
quick_df = pd.DataFrame(quick_results)

print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

# Best for all trades
best_all = all_df.loc[all_df['edge_bps'].idxmax()]
print(f"\nBest stop for ALL trades: {best_all['stop_pct']:.2f}% with {best_all['edge_bps']:.2f} bps")

# Best for quick exits
best_quick = quick_df.loc[quick_df['edge_bps'].idxmax()]
print(f"Best stop for QUICK exits: {best_quick['stop_pct']:.2f}% with {best_quick['edge_bps']:.2f} bps")

# Find sweet spot - good edge with low stop rate
print("\nSweet spots (>0.5 bps edge, <20% stop rate):")
sweet_spots = quick_df[(quick_df['edge_bps'] > 0.5) & (quick_df['stop_rate'] < 0.20)]
if len(sweet_spots) > 0:
    for _, row in sweet_spots.iterrows():
        print(f"  {row['stop_pct']:.2f}%: {row['edge_bps']:.2f} bps, {row['stop_rate']:.1%} stopped, "
              f"{row['trades_per_day']:.1f} tpd")

# For the specific "Quick exits (<5 bars)" strategy
print(f"\n\nFor your Quick Exits (<5 bars) strategy:")
print(f"{'Stop %':<8} {'Edge':<12} {'After Cost':<12} {'Annual Return':<15}")
print("-" * 55)

# Calculate annualized returns for key stop levels
key_stops = [0.10, 0.15, 0.20, 0.25, 0.30]
trading_days_per_year = 252

for stop_pct in key_stops:
    row = quick_df[quick_df['stop_pct'] == stop_pct]
    if len(row) > 0:
        edge_bps = row.iloc[0]['edge_bps']
        tpd = row.iloc[0]['trades_per_day']
        
        # Net edge after 1 bps cost
        net_edge_bps = edge_bps - 1.0
        
        # Annual return calculation
        # (net bps per trade / 10000) * trades per day * trading days
        annual_return = (net_edge_bps / 10000) * tpd * trading_days_per_year * 100
        
        print(f"{stop_pct:>6.2f}% {edge_bps:>10.2f} bps {net_edge_bps:>10.2f} bps "
              f"{annual_return:>13.2f}%")