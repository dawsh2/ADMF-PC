"""Analyze Keltner Bands with various stop loss strategies"""
import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path("workspaces/signal_generation_d5807cc2")
signal_file = workspace / "traces/SPY_1m/signals/keltner_bands/SPY_compiled_strategy_0.parquet"

print("=== KELTNER BANDS WITH STOP LOSSES ===\n")

# Load signals
signals = pd.read_parquet(signal_file)

# Load SPY 1m data
spy_1m = pd.read_csv("./data/SPY.csv")
spy_1m['timestamp'] = pd.to_datetime(spy_1m['timestamp'], utc=True)
spy_1m = spy_1m.rename(columns={col: col.lower() for col in spy_1m.columns if col != 'timestamp'})
spy_subset = spy_1m.iloc[:81787].copy()

# Calculate ATR for dynamic stops
spy_subset['tr'] = np.maximum(
    spy_subset['high'] - spy_subset['low'],
    np.maximum(
        abs(spy_subset['high'] - spy_subset['close'].shift(1)),
        abs(spy_subset['low'] - spy_subset['close'].shift(1))
    )
)
spy_subset['atr_20'] = spy_subset['tr'].rolling(20).mean()

def analyze_with_stop(stop_type, stop_value):
    """Analyze performance with a specific stop loss strategy"""
    trades = []
    entry_data = None
    
    for i in range(len(signals)):
        curr = signals.iloc[i]
        
        if entry_data is None and curr['val'] != 0:
            entry_idx = curr['idx']
            if entry_idx < len(spy_subset):
                entry_bar = spy_subset.iloc[entry_idx]
                entry_data = {
                    'idx': entry_idx,
                    'price': curr['px'],
                    'signal': curr['val'],
                    'high_water': curr['px'] if curr['val'] > 0 else None,
                    'low_water': curr['px'] if curr['val'] < 0 else None,
                    'entry_atr': entry_bar.get('atr_20', 0)
                }
        
        elif entry_data is not None:
            # Check each bar between entry and current signal
            start_idx = entry_data['idx'] + 1
            end_idx = min(curr['idx'] + 1, len(spy_subset))
            
            stopped_out = False
            exit_idx = curr['idx']
            exit_price = curr['px']
            
            for check_idx in range(start_idx, end_idx):
                if check_idx >= len(spy_subset):
                    break
                    
                check_bar = spy_subset.iloc[check_idx]
                
                # Update trailing stop levels
                if entry_data['signal'] > 0:  # Long
                    if entry_data['high_water'] is None or check_bar['high'] > entry_data['high_water']:
                        entry_data['high_water'] = check_bar['high']
                else:  # Short
                    if entry_data['low_water'] is None or check_bar['low'] < entry_data['low_water']:
                        entry_data['low_water'] = check_bar['low']
                
                # Check stop conditions
                if stop_type == 'fixed_pct':
                    # Fixed percentage stop
                    if entry_data['signal'] > 0:  # Long
                        stop_level = entry_data['price'] * (1 - stop_value)
                        if check_bar['low'] <= stop_level:
                            stopped_out = True
                            exit_price = stop_level
                            exit_idx = check_idx
                            break
                    else:  # Short
                        stop_level = entry_data['price'] * (1 + stop_value)
                        if check_bar['high'] >= stop_level:
                            stopped_out = True
                            exit_price = stop_level
                            exit_idx = check_idx
                            break
                            
                elif stop_type == 'trailing_pct':
                    # Trailing percentage stop
                    if entry_data['signal'] > 0 and entry_data['high_water']:  # Long
                        stop_level = entry_data['high_water'] * (1 - stop_value)
                        if check_bar['low'] <= stop_level:
                            stopped_out = True
                            exit_price = stop_level
                            exit_idx = check_idx
                            break
                    elif entry_data['signal'] < 0 and entry_data['low_water']:  # Short
                        stop_level = entry_data['low_water'] * (1 + stop_value)
                        if check_bar['high'] >= stop_level:
                            stopped_out = True
                            exit_price = stop_level
                            exit_idx = check_idx
                            break
                            
                elif stop_type == 'atr_stop':
                    # ATR-based stop
                    if entry_data['signal'] > 0:  # Long
                        stop_level = entry_data['price'] - (entry_data['entry_atr'] * stop_value)
                        if check_bar['low'] <= stop_level:
                            stopped_out = True
                            exit_price = stop_level
                            exit_idx = check_idx
                            break
                    else:  # Short
                        stop_level = entry_data['price'] + (entry_data['entry_atr'] * stop_value)
                        if check_bar['high'] >= stop_level:
                            stopped_out = True
                            exit_price = stop_level
                            exit_idx = check_idx
                            break
            
            # Process exit (natural signal exit or stop)
            should_exit = stopped_out or curr['val'] == 0 or np.sign(curr['val']) != np.sign(entry_data['signal'])
            
            if should_exit and entry_data['idx'] < len(spy_subset):
                pct_return = (exit_price / entry_data['price'] - 1) * entry_data['signal'] * 100
                duration = exit_idx - entry_data['idx']
                
                trade = {
                    'pct_return': pct_return,
                    'direction': 'short' if entry_data['signal'] < 0 else 'long',
                    'duration': duration,
                    'stopped_out': stopped_out,
                    'entry_idx': entry_data['idx'],
                    'exit_idx': exit_idx
                }
                trades.append(trade)
                
                # Clear or update entry
                if curr['val'] != 0 and not stopped_out:
                    entry_idx = curr['idx']
                    if entry_idx < len(spy_subset):
                        entry_bar = spy_subset.iloc[entry_idx]
                        entry_data = {
                            'idx': entry_idx,
                            'price': curr['px'],
                            'signal': curr['val'],
                            'high_water': curr['px'] if curr['val'] > 0 else None,
                            'low_water': curr['px'] if curr['val'] < 0 else None,
                            'entry_atr': entry_bar.get('atr_20', 0)
                        }
                else:
                    entry_data = None
    
    return pd.DataFrame(trades)

# Test different stop strategies
stop_strategies = [
    ('No stop (baseline)', None, None),
    ('Fixed 0.1% stop', 'fixed_pct', 0.001),
    ('Fixed 0.2% stop', 'fixed_pct', 0.002),
    ('Fixed 0.3% stop', 'fixed_pct', 0.003),
    ('Trailing 0.1% stop', 'trailing_pct', 0.001),
    ('Trailing 0.2% stop', 'trailing_pct', 0.002),
    ('1x ATR stop', 'atr_stop', 1.0),
    ('1.5x ATR stop', 'atr_stop', 1.5),
    ('2x ATR stop', 'atr_stop', 2.0),
]

total_days = 81787 / 390
results = []

for stop_name, stop_type, stop_value in stop_strategies:
    if stop_type is None:
        # Baseline - no stops
        trades_df = analyze_with_stop('no_stop', 0)
    else:
        trades_df = analyze_with_stop(stop_type, stop_value)
    
    if len(trades_df) > 0:
        avg_return = trades_df['pct_return'].mean()
        win_rate = (trades_df['pct_return'] > 0).mean()
        tpd = len(trades_df) / total_days
        
        # Separate stopped vs natural exits
        if stop_type is not None:
            stopped = trades_df[trades_df['stopped_out']]
            natural = trades_df[~trades_df['stopped_out']]
            stop_rate = len(stopped) / len(trades_df) if len(trades_df) > 0 else 0
        else:
            stop_rate = 0
            
        results.append({
            'strategy': stop_name,
            'avg_return_pct': avg_return,
            'avg_return_bps': avg_return * 100,  # Convert to actual basis points
            'win_rate': win_rate,
            'trades_per_day': tpd,
            'total_trades': len(trades_df),
            'stop_rate': stop_rate,
            'avg_duration': trades_df['duration'].mean()
        })
        
        print(f"\n{stop_name}:")
        print(f"  Average return: {avg_return:.4f}% = {avg_return * 100:.2f} basis points")
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Trades/day: {tpd:.1f}")
        print(f"  Avg duration: {trades_df['duration'].mean():.1f} bars")
        if stop_type is not None:
            print(f"  Stopped out: {stop_rate:.1%} of trades")

# Summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('avg_return_bps', ascending=False)

print("\nRanked by edge (basis points):")
print("-" * 80)
print(f"{'Strategy':<25} {'Edge (bps)':<12} {'Win Rate':<10} {'Trades/Day':<12} {'Stop Rate':<10}")
print("-" * 80)

for _, row in results_df.iterrows():
    print(f"{row['strategy']:<25} {row['avg_return_bps']:>10.2f} {row['win_rate']:>9.1%} "
          f"{row['trades_per_day']:>11.1f} {row['stop_rate']:>9.1%}")

# Check if any meet requirements
meets_requirements = results_df[(results_df['avg_return_bps'] >= 1.0) & (results_df['trades_per_day'] >= 2.0)]

print("\n" + "="*80)
if len(meets_requirements) > 0:
    print("✓ STRATEGIES MEETING YOUR REQUIREMENTS (>=1 bps, 2+ tpd):")
    for _, row in meets_requirements.iterrows():
        print(f"  {row['strategy']}: {row['avg_return_bps']:.2f} bps on {row['trades_per_day']:.1f} tpd")
else:
    print("✗ No stop loss strategy achieves >=1 bps with 2+ trades/day")
    
    # Best edge
    best_edge = results_df.iloc[0]
    print(f"\nBest edge: {best_edge['strategy']} with {best_edge['avg_return_bps']:.2f} bps")
    
    # Best with decent frequency
    good_freq = results_df[results_df['trades_per_day'] >= 10]
    if len(good_freq) > 0:
        best_freq = good_freq.iloc[0]
        print(f"Best with 10+ tpd: {best_freq['strategy']} with {best_freq['avg_return_bps']:.2f} bps on {best_freq['trades_per_day']:.1f} tpd")