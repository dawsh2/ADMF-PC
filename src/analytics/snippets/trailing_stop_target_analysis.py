# Trailing Stop/Target Analysis for Bollinger Strategies
# Tests various trailing stop and trailing target configurations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250625_173629')
SIGNAL_DIR = RESULTS_DIR / 'traces/signals/bollinger_bands'
DATA_DIR = Path('/Users/daws/ADMF-PC/data')

# Trailing configurations to test
TRAILING_CONFIGS = [
    # (initial_stop%, trail_stop%, initial_target%, trail_target%, trail_target_after%)
    # Fixed stops with trailing targets
    (0.075, None, 0.10, 0.05, 0.05),   # Fixed 0.075% stop, target trails by 0.05% after 0.05% profit
    (0.075, None, 0.10, 0.075, 0.075), # Fixed 0.075% stop, target trails by 0.075% after 0.075% profit
    (0.10, None, 0.15, 0.10, 0.10),    # Fixed 0.10% stop, target trails by 0.10% after 0.10% profit
    
    # Trailing stops with fixed targets
    (0.10, 0.05, 0.15, None, None),    # Initial 0.10% stop, trails by 0.05%, fixed 0.15% target
    (0.10, 0.075, 0.20, None, None),   # Initial 0.10% stop, trails by 0.075%, fixed 0.20% target
    (0.15, 0.10, 0.30, None, None),    # Initial 0.15% stop, trails by 0.10%, fixed 0.30% target
    
    # Both trailing
    (0.10, 0.05, 0.15, 0.05, 0.10),    # Both trail by 0.05%
    (0.10, 0.075, 0.20, 0.10, 0.10),   # Stop trails 0.075%, target trails 0.10%
    (0.15, 0.10, 0.30, 0.15, 0.15),    # Both trail aggressively
    
    # Baseline comparisons
    (0.075, None, 0.10, None, None),   # Fixed stop/target (proven optimal)
    (0, None, 0, None, None),           # No stop/target
]

print("📊 Trailing Stop/Target Analysis for Bollinger Strategies")
print("=" * 80)

# Load market data
print("Loading market data...")
market_data = None

# Look for 5-minute data
for pattern in ['*5m*.csv', '*5min*.csv', '*_5m.csv', 'SPY_5m.csv']:
    files = list(DATA_DIR.glob(pattern))
    if files:
        market_data = pd.read_csv(files[0])
        print(f"Loaded {files[0].name}")
        break

if market_data is None:
    print("❌ Could not find 5-minute market data!")
    import sys
    sys.exit()

# Clean timestamps
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], utc=True).dt.tz_localize(None)

# Determine column names
close_col = 'Close' if 'Close' in market_data.columns else 'close'
low_col = 'Low' if 'Low' in market_data.columns else 'low'
high_col = 'High' if 'High' in market_data.columns else 'high'

print(f"Market data: {len(market_data)} rows from {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")

def extract_trades_with_trailing(signal_file, market_data, initial_stop_pct, trail_stop_pct, 
                                initial_target_pct, trail_target_pct, trail_target_after_pct,
                                execution_cost_bps=1.0):
    """
    Extract trades with trailing stop/target logic
    
    Parameters:
    - initial_stop_pct: Initial stop loss percentage
    - trail_stop_pct: How much to trail the stop (None for fixed stop)
    - initial_target_pct: Initial profit target percentage
    - trail_target_pct: How much to trail the target (None for fixed target)
    - trail_target_after_pct: Start trailing target after this profit % is reached
    """
    
    # Load signals
    df = pd.read_parquet(signal_file)
    df['ts'] = pd.to_datetime(df['ts'])
    if hasattr(df['ts'].dtype, 'tz'):
        df['ts'] = df['ts'].dt.tz_localize(None)
    
    df = df.sort_values('ts')
    
    trades = []
    current_pos = 0
    entry = None
    
    for idx, row in df.iterrows():
        signal = row['val']
        
        if current_pos == 0 and signal != 0:
            # Entry
            current_pos = signal
            entry = {
                'time': row['ts'], 
                'price': row['px'], 
                'direction': signal
            }
            
        elif current_pos != 0 and signal != current_pos:
            # Exit - but check for trailing stop/target first
            if entry:
                # Find market data between entry and current signal
                mask = (market_data['timestamp'] >= entry['time']) & (market_data['timestamp'] <= row['ts'])
                trade_bars = market_data[mask]
                
                if len(trade_bars) == 0:
                    continue
                
                exit_price = row['px']
                exit_type = 'signal'
                exit_time = row['ts']
                
                # Initialize stop and target prices
                entry_price = entry['price']
                
                if entry['direction'] > 0:  # Long
                    stop_price = entry_price * (1 - initial_stop_pct/100) if initial_stop_pct > 0 else 0
                    target_price = entry_price * (1 + initial_target_pct/100) if initial_target_pct > 0 else float('inf')
                    highest_price = entry_price
                    target_activated = False
                    
                    for bar_idx, bar in trade_bars.iterrows():
                        current_high = bar[high_col]
                        current_low = bar[low_col]
                        
                        # Update highest price
                        if current_high > highest_price:
                            highest_price = current_high
                            
                            # Trail stop if configured
                            if trail_stop_pct is not None and initial_stop_pct > 0:
                                new_stop = highest_price * (1 - trail_stop_pct/100)
                                stop_price = max(stop_price, new_stop)
                            
                            # Trail target if configured and profit threshold reached
                            if trail_target_pct is not None and initial_target_pct > 0:
                                profit_pct = (highest_price - entry_price) / entry_price * 100
                                if profit_pct >= trail_target_after_pct:
                                    target_activated = True
                                    new_target = highest_price * (1 + trail_target_pct/100)
                                    target_price = max(target_price, new_target)
                        
                        # Check exits
                        if initial_stop_pct > 0 and current_low <= stop_price:
                            exit_price = stop_price
                            exit_type = 'stop' if stop_price <= entry_price else 'trailing_stop'
                            exit_time = bar['timestamp']
                            break
                        elif initial_target_pct > 0 and current_high >= target_price:
                            exit_price = target_price
                            exit_type = 'target' if not target_activated else 'trailing_target'
                            exit_time = bar['timestamp']
                            break
                            
                else:  # Short
                    stop_price = entry_price * (1 + initial_stop_pct/100) if initial_stop_pct > 0 else float('inf')
                    target_price = entry_price * (1 - initial_target_pct/100) if initial_target_pct > 0 else 0
                    lowest_price = entry_price
                    target_activated = False
                    
                    for bar_idx, bar in trade_bars.iterrows():
                        current_high = bar[high_col]
                        current_low = bar[low_col]
                        
                        # Update lowest price
                        if current_low < lowest_price:
                            lowest_price = current_low
                            
                            # Trail stop if configured
                            if trail_stop_pct is not None and initial_stop_pct > 0:
                                new_stop = lowest_price * (1 + trail_stop_pct/100)
                                stop_price = min(stop_price, new_stop)
                            
                            # Trail target if configured and profit threshold reached
                            if trail_target_pct is not None and initial_target_pct > 0:
                                profit_pct = (entry_price - lowest_price) / entry_price * 100
                                if profit_pct >= trail_target_after_pct:
                                    target_activated = True
                                    new_target = lowest_price * (1 - trail_target_pct/100)
                                    target_price = min(target_price, new_target)
                        
                        # Check exits
                        if initial_stop_pct > 0 and current_high >= stop_price:
                            exit_price = stop_price
                            exit_type = 'stop' if stop_price >= entry_price else 'trailing_stop'
                            exit_time = bar['timestamp']
                            break
                        elif initial_target_pct > 0 and current_low <= target_price:
                            exit_price = target_price
                            exit_type = 'target' if not target_activated else 'trailing_target'
                            exit_time = bar['timestamp']
                            break
                
                # Calculate return
                if entry['direction'] > 0:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - (execution_cost_bps * 2 / 10000)
                
                trades.append({
                    'entry_time': entry['time'],
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': entry['direction'],
                    'raw_return': raw_return,
                    'net_return': net_return,
                    'exit_type': exit_type,
                    'duration_min': (exit_time - entry['time']).total_seconds() / 60,
                    'max_profit_pct': ((highest_price if entry['direction'] > 0 else entry_price - lowest_price) - entry_price) / entry_price * 100 if entry['direction'] > 0 else (entry_price - lowest_price) / entry_price * 100
                })
            
            # Update position
            current_pos = signal
            if signal != 0:
                entry = {'time': row['ts'], 'price': row['px'], 'direction': signal}
            else:
                entry = None
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

# Load strategy index
strategy_index = pd.read_parquet(RESULTS_DIR / 'strategy_index.parquet')

# Find top strategies by signal count
print("\nFinding strategies with sufficient signals...")
strategy_files = []
all_signal_counts = []

for signal_file in SIGNAL_DIR.glob('*.parquet'):
    df = pd.read_parquet(signal_file)
    non_zero = (df['val'] != 0).sum()
    all_signal_counts.append(non_zero)
    
    # Lower threshold for test set (4k bars of 5m data)
    if non_zero > 100:  # Adjusted for test set
        strategy_num = int(signal_file.stem.split('_')[-1])
        if strategy_num < len(strategy_index):
            params = strategy_index.iloc[strategy_num]
            strategy_files.append({
                'file': signal_file,
                'period': params.get('period', 'N/A'),
                'std_dev': params.get('std_dev', 'N/A'),
                'signal_count': non_zero
            })

# Debug info
print(f"\nSignal count stats: min={min(all_signal_counts)}, max={max(all_signal_counts)}, mean={np.mean(all_signal_counts):.0f}")
print(f"Strategies with >50 signals: {sum(1 for s in all_signal_counts if s > 50)}")
print(f"Strategies with >100 signals: {sum(1 for s in all_signal_counts if s > 100)}")
print(f"Strategies with >200 signals: {sum(1 for s in all_signal_counts if s > 200)}")

# Sort by signal count and take top 5
strategy_files = sorted(strategy_files, key=lambda x: x['signal_count'], reverse=True)[:5]
print(f"\nTesting top {len(strategy_files)} strategies (with >{100 if len(strategy_files) > 0 else 0} signals)")

# Test all configurations
results = []

for strategy_idx, strategy in enumerate(strategy_files):
    print(f"\nTesting strategy {strategy_idx+1}/{len(strategy_files)}: "
          f"P{strategy['period']}_S{strategy['std_dev']} ({strategy['signal_count']} signals)")
    
    for config_idx, (initial_stop, trail_stop, initial_target, trail_target, trail_target_after) in enumerate(TRAILING_CONFIGS):
        print(f"\r  Config {config_idx+1}/{len(TRAILING_CONFIGS)}...", end='', flush=True)
        
        try:
            trades_df = extract_trades_with_trailing(
                strategy['file'], 
                market_data,
                initial_stop,
                trail_stop,
                initial_target,
                trail_target,
                trail_target_after,
                execution_cost_bps=1.0
            )
            
            if len(trades_df) > 10:
                # Calculate metrics
                total_return = (1 + trades_df['net_return']).prod() - 1
                win_rate = (trades_df['net_return'] > 0).mean()
                avg_return = trades_df['net_return'].mean()
                
                # Exit type breakdown
                exit_counts = trades_df['exit_type'].value_counts()
                
                # Sharpe ratio
                if trades_df['net_return'].std() > 0:
                    trading_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days or 1
                    trades_per_day = len(trades_df) / trading_days
                    sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * trades_per_day)
                else:
                    sharpe = 0
                
                # Config description
                config_desc = f"Stop: {initial_stop}%"
                if trail_stop:
                    config_desc += f" (trail {trail_stop}%)"
                config_desc += f", Target: {initial_target}%"
                if trail_target:
                    config_desc += f" (trail {trail_target}% after {trail_target_after}%)"
                
                results.append({
                    'strategy': f"P{strategy['period']}_S{strategy['std_dev']}",
                    'config': config_desc,
                    'initial_stop': initial_stop,
                    'trail_stop': trail_stop or 0,
                    'initial_target': initial_target,
                    'trail_target': trail_target or 0,
                    'num_trades': len(trades_df),
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'avg_duration': trades_df['duration_min'].mean(),
                    'stop_exits': exit_counts.get('stop', 0),
                    'trailing_stop_exits': exit_counts.get('trailing_stop', 0),
                    'target_exits': exit_counts.get('target', 0),
                    'trailing_target_exits': exit_counts.get('trailing_target', 0),
                    'signal_exits': exit_counts.get('signal', 0),
                    'avg_max_profit': trades_df['max_profit_pct'].mean()
                })
                
        except Exception as e:
            print(f"\n  Error: {e}")
            continue

print("\n\nAnalysis complete!")

if results:
    results_df = pd.DataFrame(results)
    
    # Find best configurations
    print("\n" + "=" * 80)
    print("🎯 TOP TRAILING CONFIGURATIONS BY SHARPE RATIO")
    print("=" * 80)
    
    top_configs = results_df.nlargest(15, 'sharpe_ratio')
    for idx, row in top_configs.iterrows():
        print(f"\n{row['strategy']} - {row['config']}")
        print(f"  Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']*100:.2f}%, "
              f"Win Rate: {row['win_rate']*100:.1f}%")
        print(f"  Exits: Stop={row['stop_exits']}, Trail Stop={row['trailing_stop_exits']}, "
              f"Target={row['target_exits']}, Trail Target={row['trailing_target_exits']}, "
              f"Signal={row['signal_exits']}")
    
    # Compare trailing vs fixed
    print("\n" + "=" * 80)
    print("📊 TRAILING VS FIXED COMPARISON")
    print("=" * 80)
    
    # Fixed baseline (0.075% stop, 0.1% target)
    fixed_baseline = results_df[(results_df['initial_stop'] == 0.075) & 
                                (results_df['trail_stop'] == 0) & 
                                (results_df['initial_target'] == 0.1) & 
                                (results_df['trail_target'] == 0)]
    
    if len(fixed_baseline) > 0:
        avg_fixed_sharpe = fixed_baseline['sharpe_ratio'].mean()
        avg_fixed_return = fixed_baseline['total_return'].mean()
        
        print(f"\nFixed Baseline (0.075% stop, 0.1% target):")
        print(f"  Avg Sharpe: {avg_fixed_sharpe:.2f}")
        print(f"  Avg Return: {avg_fixed_return*100:.2f}%")
        
        # Best trailing stop (fixed target)
        trailing_stop_configs = results_df[(results_df['trail_stop'] > 0) & (results_df['trail_target'] == 0)]
        if len(trailing_stop_configs) > 0:
            best_trail_stop = trailing_stop_configs.nlargest(1, 'sharpe_ratio').iloc[0]
            print(f"\nBest Trailing Stop Config:")
            print(f"  {best_trail_stop['config']}")
            print(f"  Sharpe: {best_trail_stop['sharpe_ratio']:.2f} "
                  f"(+{(best_trail_stop['sharpe_ratio']/avg_fixed_sharpe - 1)*100:.1f}% vs fixed)")
            print(f"  Return: {best_trail_stop['total_return']*100:.2f}%")
        
        # Best trailing target (fixed stop)
        trailing_target_configs = results_df[(results_df['trail_stop'] == 0) & (results_df['trail_target'] > 0)]
        if len(trailing_target_configs) > 0:
            best_trail_target = trailing_target_configs.nlargest(1, 'sharpe_ratio').iloc[0]
            print(f"\nBest Trailing Target Config:")
            print(f"  {best_trail_target['config']}")
            print(f"  Sharpe: {best_trail_target['sharpe_ratio']:.2f} "
                  f"(+{(best_trail_target['sharpe_ratio']/avg_fixed_sharpe - 1)*100:.1f}% vs fixed)")
            print(f"  Return: {best_trail_target['total_return']*100:.2f}%")
        
        # Best both trailing
        both_trailing = results_df[(results_df['trail_stop'] > 0) & (results_df['trail_target'] > 0)]
        if len(both_trailing) > 0:
            best_both = both_trailing.nlargest(1, 'sharpe_ratio').iloc[0]
            print(f"\nBest Both Trailing Config:")
            print(f"  {best_both['config']}")
            print(f"  Sharpe: {best_both['sharpe_ratio']:.2f} "
                  f"(+{(best_both['sharpe_ratio']/avg_fixed_sharpe - 1)*100:.1f}% vs fixed)")
            print(f"  Return: {best_both['total_return']*100:.2f}%")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sharpe comparison: Fixed vs Trailing
    ax = axes[0, 0]
    categories = ['Fixed\nStop/Target', 'Trailing\nStop Only', 'Trailing\nTarget Only', 'Both\nTrailing']
    avg_sharpes = []
    
    for cat, condition in [
        ('Fixed', (results_df['trail_stop'] == 0) & (results_df['trail_target'] == 0)),
        ('Trail Stop', (results_df['trail_stop'] > 0) & (results_df['trail_target'] == 0)),
        ('Trail Target', (results_df['trail_stop'] == 0) & (results_df['trail_target'] > 0)),
        ('Both', (results_df['trail_stop'] > 0) & (results_df['trail_target'] > 0))
    ]:
        subset = results_df[condition]
        if len(subset) > 0:
            avg_sharpes.append(subset['sharpe_ratio'].mean())
        else:
            avg_sharpes.append(0)
    
    ax.bar(categories, avg_sharpes)
    ax.set_ylabel('Average Sharpe Ratio')
    ax.set_title('Performance by Stop/Target Type')
    ax.grid(True, alpha=0.3)
    
    # 2. Win rate comparison
    ax = axes[0, 1]
    win_rates = []
    for cat, condition in [
        ('Fixed', (results_df['trail_stop'] == 0) & (results_df['trail_target'] == 0)),
        ('Trail Stop', (results_df['trail_stop'] > 0) & (results_df['trail_target'] == 0)),
        ('Trail Target', (results_df['trail_stop'] == 0) & (results_df['trail_target'] > 0)),
        ('Both', (results_df['trail_stop'] > 0) & (results_df['trail_target'] > 0))
    ]:
        subset = results_df[condition]
        if len(subset) > 0:
            win_rates.append(subset['win_rate'].mean() * 100)
        else:
            win_rates.append(0)
    
    ax.bar(categories, win_rates)
    ax.set_ylabel('Average Win Rate %')
    ax.set_title('Win Rate by Stop/Target Type')
    ax.grid(True, alpha=0.3)
    
    # 3. Exit type distribution for best config
    ax = axes[1, 0]
    best_config = results_df.nlargest(1, 'sharpe_ratio').iloc[0]
    exit_types = ['Stop', 'Trail Stop', 'Target', 'Trail Target', 'Signal']
    exit_counts = [
        best_config['stop_exits'],
        best_config['trailing_stop_exits'],
        best_config['target_exits'],
        best_config['trailing_target_exits'],
        best_config['signal_exits']
    ]
    
    ax.pie([c for c in exit_counts if c > 0], 
           labels=[t for t, c in zip(exit_types, exit_counts) if c > 0],
           autopct='%1.1f%%')
    ax.set_title(f'Exit Types - Best Config\n{best_config["config"]}')
    
    # 4. Scatter: Sharpe vs Max Profit Utilization
    ax = axes[1, 1]
    ax.scatter(results_df['avg_max_profit'], results_df['sharpe_ratio'], alpha=0.6)
    ax.set_xlabel('Average Max Profit %')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe vs Profit Potential Captured')
    ax.grid(True, alpha=0.3)
    
    # Highlight different types
    for config_type, color, marker in [
        ((results_df['trail_stop'] == 0) & (results_df['trail_target'] == 0), 'blue', 'o'),
        ((results_df['trail_stop'] > 0) & (results_df['trail_target'] == 0), 'green', '^'),
        ((results_df['trail_stop'] == 0) & (results_df['trail_target'] > 0), 'orange', 's'),
        ((results_df['trail_stop'] > 0) & (results_df['trail_target'] > 0), 'red', '*')
    ]:
        subset = results_df[config_type]
        if len(subset) > 0:
            ax.scatter(subset['avg_max_profit'], subset['sharpe_ratio'], 
                      color=color, marker=marker, s=100, alpha=0.8)
    
    ax.legend(['All', 'Fixed', 'Trail Stop', 'Trail Target', 'Both Trail'])
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    output_path = RESULTS_DIR / 'trailing_stop_target_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Summary recommendations
    print("\n" + "=" * 80)
    print("💡 KEY FINDINGS & RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. TRAILING STOPS:")
    trailing_stops = results_df[results_df['trail_stop'] > 0]
    if len(trailing_stops) > 0:
        avg_trail_stop_sharpe = trailing_stops['sharpe_ratio'].mean()
        print(f"   - Average Sharpe with trailing stops: {avg_trail_stop_sharpe:.2f}")
        print(f"   - Most effective trail distance: {trailing_stops.nlargest(5, 'sharpe_ratio')['trail_stop'].mode().values[0]:.3f}%")
    
    print("\n2. TRAILING TARGETS:")
    trailing_targets = results_df[results_df['trail_target'] > 0]
    if len(trailing_targets) > 0:
        avg_trail_target_sharpe = trailing_targets['sharpe_ratio'].mean()
        print(f"   - Average Sharpe with trailing targets: {avg_trail_target_sharpe:.2f}")
        print(f"   - Optimal activation level: Start trailing after {trailing_targets.nlargest(5, 'sharpe_ratio')['initial_target'].mode().values[0]:.3f}% profit")
    
    print("\n3. OPTIMAL CONFIGURATION:")
    best = results_df.nlargest(1, 'sharpe_ratio').iloc[0]
    print(f"   - {best['config']}")
    print(f"   - Sharpe: {best['sharpe_ratio']:.2f}, Return: {best['total_return']*100:.2f}%")
    print(f"   - This config captures {best['avg_max_profit']:.2f}% average max profit per trade")
    
else:
    print("\n❌ No valid results generated!")

print("\n" + "=" * 80)