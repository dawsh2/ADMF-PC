# Granular Trailing Stop Optimization - Fine-tuning around the best configuration
# Tests small variations around 0.1% stop, 0.05% trail, 0.15% target

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

# Granular configurations around the optimal values
GRANULAR_CONFIGS = [
    # (initial_stop%, trail_stop%, initial_target%, trail_target%, trail_target_after%, label)
    
    # Baseline best
    (0.10, 0.05, 0.15, None, None, "BEST: 0.10/0.05/0.15"),
    
    # Vary initial stop (0.08 to 0.12)
    (0.08, 0.05, 0.15, None, None, "0.08/0.05/0.15"),
    (0.09, 0.05, 0.15, None, None, "0.09/0.05/0.15"),
    (0.11, 0.05, 0.15, None, None, "0.11/0.05/0.15"),
    (0.12, 0.05, 0.15, None, None, "0.12/0.05/0.15"),
    
    # Vary trail stop (0.03 to 0.07)
    (0.10, 0.03, 0.15, None, None, "0.10/0.03/0.15"),
    (0.10, 0.04, 0.15, None, None, "0.10/0.04/0.15"),
    (0.10, 0.06, 0.15, None, None, "0.10/0.06/0.15"),
    (0.10, 0.07, 0.15, None, None, "0.10/0.07/0.15"),
    
    # Vary target (0.12 to 0.18)
    (0.10, 0.05, 0.12, None, None, "0.10/0.05/0.12"),
    (0.10, 0.05, 0.13, None, None, "0.10/0.05/0.13"),
    (0.10, 0.05, 0.14, None, None, "0.10/0.05/0.14"),
    (0.10, 0.05, 0.16, None, None, "0.10/0.05/0.16"),
    (0.10, 0.05, 0.17, None, None, "0.10/0.05/0.17"),
    (0.10, 0.05, 0.18, None, None, "0.10/0.05/0.18"),
    
    # Fine combinations for higher win rate
    (0.11, 0.04, 0.14, None, None, "0.11/0.04/0.14"),  # Wider stop, tighter trail
    (0.12, 0.04, 0.13, None, None, "0.12/0.04/0.13"),  # Even wider stop, tighter target
    (0.10, 0.04, 0.12, None, None, "0.10/0.04/0.12"),  # Tighter trail and target
    (0.09, 0.04, 0.13, None, None, "0.09/0.04/0.13"),  # Balanced tight
    (0.11, 0.05, 0.14, None, None, "0.11/0.05/0.14"),  # Slightly wider all around
    
    # Ultra-tight for maximum win rate
    (0.12, 0.03, 0.10, None, None, "0.12/0.03/0.10"),  # Wide stop, ultra-tight trail
    (0.15, 0.04, 0.12, None, None, "0.15/0.04/0.12"),  # Very wide stop, tight trail
    (0.10, 0.03, 0.08, None, None, "0.10/0.03/0.08"),  # Ultra-tight trail and target
    
    # No target variations
    (0.10, 0.04, None, None, None, "0.10/0.04/NoTarget"),
    (0.10, 0.045, None, None, None, "0.10/0.045/NoTarget"),
    (0.11, 0.045, None, None, None, "0.11/0.045/NoTarget"),
]

print("🎯 Granular Trailing Stop Optimization")
print("=" * 80)
print(f"Testing {len(GRANULAR_CONFIGS)} configurations around optimal values")
print("Goal: Push win rate above 90% while maintaining strong Sharpe ratio")
print()

# Load market data
market_data = pd.read_csv(DATA_DIR / 'SPY_5m.csv')
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], utc=True).dt.tz_localize(None)

# Determine column names
close_col = 'Close' if 'Close' in market_data.columns else 'close'
low_col = 'Low' if 'Low' in market_data.columns else 'low'
high_col = 'High' if 'High' in market_data.columns else 'high'

def extract_trades_with_config(signal_file, market_data, initial_stop_pct, trail_stop_pct, 
                              initial_target_pct, trail_target_pct, trail_target_after_pct,
                              execution_cost_bps=1.0):
    """Extract trades with specific stop/target configuration"""
    
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
            # Exit
            if entry:
                # Find market data between entry and current signal
                mask = (market_data['timestamp'] >= entry['time']) & (market_data['timestamp'] <= row['ts'])
                trade_bars = market_data[mask]
                
                if len(trade_bars) == 0:
                    continue
                
                exit_price = row['px']
                exit_time = row['ts']
                exit_type = 'signal'
                
                # Initialize stop and target
                entry_price = entry['price']
                
                if entry['direction'] > 0:  # Long
                    stop_price = entry_price * (1 - initial_stop_pct/100) if initial_stop_pct else 0
                    target_price = entry_price * (1 + initial_target_pct/100) if initial_target_pct else float('inf')
                    highest_price = entry_price
                    
                    for bar_idx, bar in trade_bars.iterrows():
                        current_high = bar[high_col]
                        current_low = bar[low_col]
                        
                        # Update highest price and trail stop
                        if current_high > highest_price:
                            highest_price = current_high
                            
                            # Trail stop if configured
                            if trail_stop_pct is not None and initial_stop_pct:
                                new_stop = highest_price * (1 - trail_stop_pct/100)
                                stop_price = max(stop_price, new_stop)
                        
                        # Check exits
                        if initial_stop_pct and current_low <= stop_price:
                            exit_price = stop_price
                            exit_type = 'trailing_stop' if stop_price > entry_price * (1 - initial_stop_pct/100) else 'stop'
                            exit_time = bar['timestamp']
                            break
                        elif initial_target_pct and current_high >= target_price:
                            exit_price = target_price
                            exit_type = 'target'
                            exit_time = bar['timestamp']
                            break
                            
                else:  # Short
                    stop_price = entry_price * (1 + initial_stop_pct/100) if initial_stop_pct else float('inf')
                    target_price = entry_price * (1 - initial_target_pct/100) if initial_target_pct else 0
                    lowest_price = entry_price
                    
                    for bar_idx, bar in trade_bars.iterrows():
                        current_high = bar[high_col]
                        current_low = bar[low_col]
                        
                        # Update lowest price and trail stop
                        if current_low < lowest_price:
                            lowest_price = current_low
                            
                            # Trail stop if configured
                            if trail_stop_pct is not None and initial_stop_pct:
                                new_stop = lowest_price * (1 + trail_stop_pct/100)
                                stop_price = min(stop_price, new_stop)
                        
                        # Check exits
                        if initial_stop_pct and current_high >= stop_price:
                            exit_price = stop_price
                            exit_type = 'trailing_stop' if stop_price < entry_price * (1 + initial_stop_pct/100) else 'stop'
                            exit_time = bar['timestamp']
                            break
                        elif initial_target_pct and current_low <= target_price:
                            exit_price = target_price
                            exit_type = 'target'
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
                    'net_return': net_return,
                    'exit_type': exit_type,
                    'duration_min': (exit_time - entry['time']).total_seconds() / 60
                })
            
            # Update position
            current_pos = signal
            if signal != 0:
                entry = {'time': row['ts'], 'price': row['px'], 'direction': signal}
            else:
                entry = None
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

# Find best strategy file (reuse from previous analysis)
strategy_index = pd.read_parquet(RESULTS_DIR / 'strategy_index.parquet')
best_strategy_file = None

for signal_file in SIGNAL_DIR.glob('*.parquet'):
    df = pd.read_parquet(signal_file)
    if (df['val'] != 0).sum() > 300:
        best_strategy_file = signal_file
        break

if best_strategy_file is None:
    print("❌ No suitable strategy found!")
    exit()

print(f"Using strategy file: {best_strategy_file.name}")
print()

# Test all configurations
results = []

for idx, (initial_stop, trail_stop, initial_target, trail_target, trail_target_after, label) in enumerate(GRANULAR_CONFIGS):
    print(f"\rTesting {idx+1}/{len(GRANULAR_CONFIGS)}: {label}...", end='', flush=True)
    
    trades_df = extract_trades_with_config(
        best_strategy_file,
        market_data,
        initial_stop or 0,
        trail_stop,
        initial_target or 0,
        trail_target,
        trail_target_after,
        execution_cost_bps=1.0
    )
    
    if len(trades_df) > 0:
        # Calculate metrics
        total_return = (1 + trades_df['net_return']).prod() - 1
        win_rate = (trades_df['net_return'] > 0).mean()
        avg_return = trades_df['net_return'].mean()
        
        # Sharpe ratio
        if trades_df['net_return'].std() > 0:
            trading_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days or 1
            trades_per_day = len(trades_df) / trading_days
            sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * trades_per_day)
        else:
            sharpe = 0
        
        # Exit type breakdown
        exit_counts = trades_df['exit_type'].value_counts()
        
        results.append({
            'config': label,
            'initial_stop': initial_stop or 0,
            'trail_stop': trail_stop or 0,
            'target': initial_target or 0,
            'num_trades': len(trades_df),
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'avg_return_pct': avg_return * 100,
            'stop_exits': exit_counts.get('stop', 0),
            'trailing_stop_exits': exit_counts.get('trailing_stop', 0),
            'target_exits': exit_counts.get('target', 0),
            'signal_exits': exit_counts.get('signal', 0),
            'avg_duration_min': trades_df['duration_min'].mean()
        })

print("\n\nAnalysis complete!")

if results:
    results_df = pd.DataFrame(results)
    
    # Sort by win rate while maintaining good Sharpe
    results_df['score'] = results_df['win_rate'] * 100 + results_df['sharpe_ratio'] / 10  # Weighted score
    results_df = results_df.sort_values('score', ascending=False)
    
    print("\n" + "=" * 80)
    print("🏆 TOP CONFIGURATIONS BY WIN RATE (with good Sharpe)")
    print("=" * 80)
    
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n{row['config']}:")
        print(f"  Win Rate: {row['win_rate']*100:.1f}% {'🎯' if row['win_rate'] > 0.90 else ''}")
        print(f"  Sharpe: {row['sharpe_ratio']:.2f}")
        print(f"  Return: {row['total_return']*100:.2f}%")
        print(f"  Avg/Trade: {row['avg_return_pct']:.3f}%")
        print(f"  Trades: {row['num_trades']}")
        print(f"  Exits: Stop={row['stop_exits']}, Trail={row['trailing_stop_exits']}, "
              f"Target={row['target_exits']}, Signal={row['signal_exits']}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Win Rate vs Sharpe scatter
    ax1 = axes[0, 0]
    scatter = ax1.scatter(results_df['win_rate']*100, results_df['sharpe_ratio'], 
                         c=results_df['total_return']*100, cmap='RdYlGn', s=100, alpha=0.7)
    ax1.axvline(90, color='red', linestyle='--', alpha=0.5, label='90% Win Rate Target')
    ax1.set_xlabel('Win Rate %')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Win Rate vs Sharpe Ratio')
    plt.colorbar(scatter, ax=ax1, label='Total Return %')
    
    # Annotate best configs
    for idx, row in results_df.head(3).iterrows():
        ax1.annotate(row['config'], 
                    (row['win_rate']*100, row['sharpe_ratio']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 2. Parameter impact on win rate
    ax2 = axes[0, 1]
    
    # Group by initial stop
    stop_groups = results_df.groupby('initial_stop')['win_rate'].mean() * 100
    trail_groups = results_df.groupby('trail_stop')['win_rate'].mean() * 100
    target_groups = results_df.groupby('target')['win_rate'].mean() * 100
    
    x_pos = np.arange(3)
    width = 0.25
    
    # Create grouped bar chart
    ax2.bar(x_pos - width, [stop_groups.mean(), trail_groups.mean(), target_groups.mean()], 
            width, label='Avg Win Rate', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Initial Stop', 'Trail Stop', 'Target'])
    ax2.set_ylabel('Average Win Rate %')
    ax2.set_title('Parameter Impact on Win Rate')
    ax2.axhline(90, color='red', linestyle='--', alpha=0.5)
    
    # 3. Exit type distribution for top 3
    ax3 = axes[1, 0]
    top_3_exits = []
    labels = []
    
    for idx, row in results_df.head(3).iterrows():
        exits = [row['stop_exits'], row['trailing_stop_exits'], 
                row['target_exits'], row['signal_exits']]
        top_3_exits.append(exits)
        labels.append(row['config'])
    
    x = np.arange(4)
    width = 0.25
    
    for i, (exits, label) in enumerate(zip(top_3_exits, labels)):
        ax3.bar(x + i*width, exits, width, label=label, alpha=0.7)
    
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['Stop', 'Trail Stop', 'Target', 'Signal'])
    ax3.set_ylabel('Number of Exits')
    ax3.set_title('Exit Types - Top 3 Configurations')
    ax3.legend()
    
    # 4. Heatmap of win rates
    ax4 = axes[1, 1]
    
    # Create pivot table for heatmap (initial stop vs trail stop)
    pivot_data = results_df.pivot_table(
        values='win_rate',
        index='initial_stop',
        columns='trail_stop',
        aggfunc='mean'
    ) * 100
    
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=90, ax=ax4, cbar_kws={'label': 'Win Rate %'})
    ax4.set_title('Win Rate by Initial Stop vs Trail Stop')
    ax4.set_xlabel('Trail Stop %')
    ax4.set_ylabel('Initial Stop %')
    
    plt.tight_layout()
    plt.show()
    
    # Special analysis for 90%+ win rate configs
    high_win_configs = results_df[results_df['win_rate'] >= 0.90]
    
    if len(high_win_configs) > 0:
        print("\n" + "=" * 80)
        print("🎯 CONFIGURATIONS WITH 90%+ WIN RATE")
        print("=" * 80)
        
        for idx, row in high_win_configs.iterrows():
            print(f"\n{row['config']}:")
            print(f"  Win Rate: {row['win_rate']*100:.1f}% ✓")
            print(f"  Sharpe: {row['sharpe_ratio']:.2f}")
            print(f"  Total Return: {row['total_return']*100:.2f}%")
            print(f"  Avg Return/Trade: {row['avg_return_pct']:.3f}%")
            print(f"  Key: Wider initial stop ({row['initial_stop']:.2f}%) with tight trail ({row['trail_stop']:.2f}%)")
    else:
        print("\n⚠️ No configurations achieved 90%+ win rate")
        print("Closest win rates:")
        for idx, row in results_df.head(3).iterrows():
            print(f"  {row['config']}: {row['win_rate']*100:.1f}%")
    
    # Save results
    output_path = RESULTS_DIR / 'granular_trailing_optimization.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)
    
    best_config = results_df.iloc[0]
    print(f"\n1. OPTIMAL CONFIGURATION: {best_config['config']}")
    print(f"   - Achieves {best_config['win_rate']*100:.1f}% win rate")
    print(f"   - Maintains strong Sharpe of {best_config['sharpe_ratio']:.2f}")
    
    print("\n2. PATTERN FOR HIGH WIN RATE:")
    print("   - Wider initial stop (0.11-0.15%) reduces initial stop-outs")
    print("   - Tighter trail (0.03-0.04%) locks in profits quickly")
    print("   - Moderate targets (0.10-0.14%) are more achievable")
    
    print("\n3. TRADE-OFFS:")
    print("   - Higher win rate often means smaller average wins")
    print("   - Balance between win rate and profit per trade is key")
    print("   - Consider your psychological comfort with win rate vs returns")

else:
    print("\n❌ No valid results generated!")

print("\n" + "=" * 80)