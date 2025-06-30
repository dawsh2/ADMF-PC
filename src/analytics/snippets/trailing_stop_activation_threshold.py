# Trailing Stop with Activation Threshold Analysis
# The trailing stop only activates after price moves favorably by X%

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

# Test configurations with activation thresholds
THRESHOLD_CONFIGS = [
    # (initial_stop%, trail_stop%, activation_threshold%, target%, label)
    # Format: initial/trail/threshold/target
    
    # Baseline (no threshold)
    (0.10, 0.05, 0, 0.15, "0.10/0.05/0.00/0.15 (Original)"),
    
    # Small thresholds (0.02% - 0.05%)
    (0.10, 0.05, 0.02, 0.15, "0.10/0.05/0.02/0.15"),
    (0.10, 0.05, 0.03, 0.15, "0.10/0.05/0.03/0.15"),
    (0.10, 0.05, 0.04, 0.15, "0.10/0.05/0.04/0.15"),
    (0.10, 0.05, 0.05, 0.15, "0.10/0.05/0.05/0.15"),
    
    # Medium thresholds (0.06% - 0.10%)
    (0.10, 0.05, 0.06, 0.15, "0.10/0.05/0.06/0.15"),
    (0.10, 0.05, 0.07, 0.15, "0.10/0.05/0.07/0.15"),
    (0.10, 0.05, 0.08, 0.15, "0.10/0.05/0.08/0.15"),
    (0.10, 0.05, 0.10, 0.15, "0.10/0.05/0.10/0.15"),
    
    # With tighter trails after activation
    (0.10, 0.03, 0.05, 0.15, "0.10/0.03/0.05/0.15"),
    (0.10, 0.04, 0.05, 0.15, "0.10/0.04/0.05/0.15"),
    
    # With wider initial stops
    (0.12, 0.05, 0.05, 0.15, "0.12/0.05/0.05/0.15"),
    (0.15, 0.05, 0.05, 0.15, "0.15/0.05/0.05/0.15"),
    
    # Different target levels
    (0.10, 0.05, 0.05, 0.10, "0.10/0.05/0.05/0.10"),
    (0.10, 0.05, 0.05, 0.12, "0.10/0.05/0.05/0.12"),
    (0.10, 0.05, 0.05, 0.20, "0.10/0.05/0.05/0.20"),
    
    # Aggressive: tight trail after good move
    (0.10, 0.02, 0.08, 0.15, "0.10/0.02/0.08/0.15"),
    (0.10, 0.03, 0.10, 0.20, "0.10/0.03/0.10/0.20"),
    
    # Conservative: wide trail after threshold
    (0.10, 0.08, 0.05, 0.15, "0.10/0.08/0.05/0.15"),
    (0.10, 0.10, 0.05, 0.15, "0.10/0.10/0.05/0.15"),
    
    # No target variations
    (0.10, 0.05, 0.05, None, "0.10/0.05/0.05/NoTarget"),
    (0.10, 0.04, 0.06, None, "0.10/0.04/0.06/NoTarget"),
    
    # Best for high win rate
    (0.12, 0.04, 0.04, 0.12, "0.12/0.04/0.04/0.12"),
    (0.15, 0.03, 0.05, 0.10, "0.15/0.03/0.05/0.10"),
]

print("🎯 Trailing Stop with Activation Threshold Analysis")
print("=" * 80)
print(f"Testing {len(THRESHOLD_CONFIGS)} configurations with activation thresholds")
print("\nHow it works:")
print("1. Initial stop loss is always active")
print("2. Trailing stop ONLY activates after price moves favorably by threshold %")
print("3. Once activated, trailing stop maintains set distance from highest price")
print()

# Load market data
market_data = pd.read_csv(DATA_DIR / 'SPY_5m.csv')
market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], utc=True).dt.tz_localize(None)

# Determine column names
close_col = 'Close' if 'Close' in market_data.columns else 'close'
low_col = 'Low' if 'Low' in market_data.columns else 'low'
high_col = 'High' if 'High' in market_data.columns else 'high'

def extract_trades_with_threshold(signal_file, market_data, initial_stop_pct, trail_stop_pct, 
                                 activation_threshold_pct, target_pct, execution_cost_bps=1.0):
    """Extract trades with trailing stop activation threshold"""
    
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
                    initial_stop_price = entry_price * (1 - initial_stop_pct/100) if initial_stop_pct else 0
                    target_price = entry_price * (1 + target_pct/100) if target_pct else float('inf')
                    highest_price = entry_price
                    trailing_activated = False
                    stop_price = initial_stop_price
                    
                    for bar_idx, bar in trade_bars.iterrows():
                        current_high = bar[high_col]
                        current_low = bar[low_col]
                        
                        # Update highest price
                        if current_high > highest_price:
                            highest_price = current_high
                            
                            # Check if we should activate trailing stop
                            profit_pct = (highest_price - entry_price) / entry_price * 100
                            
                            if not trailing_activated and profit_pct >= activation_threshold_pct:
                                trailing_activated = True
                                # Set initial trailing stop position
                                stop_price = highest_price * (1 - trail_stop_pct/100)
                            elif trailing_activated:
                                # Update trailing stop
                                new_stop = highest_price * (1 - trail_stop_pct/100)
                                stop_price = max(stop_price, new_stop)
                        
                        # Check exits
                        if initial_stop_pct and current_low <= stop_price:
                            exit_price = stop_price
                            if trailing_activated:
                                exit_type = 'trailing_stop'
                            else:
                                exit_type = 'stop'
                            exit_time = bar['timestamp']
                            break
                        elif target_pct and current_high >= target_price:
                            exit_price = target_price
                            exit_type = 'target'
                            exit_time = bar['timestamp']
                            break
                            
                else:  # Short
                    initial_stop_price = entry_price * (1 + initial_stop_pct/100) if initial_stop_pct else float('inf')
                    target_price = entry_price * (1 - target_pct/100) if target_pct else 0
                    lowest_price = entry_price
                    trailing_activated = False
                    stop_price = initial_stop_price
                    
                    for bar_idx, bar in trade_bars.iterrows():
                        current_high = bar[high_col]
                        current_low = bar[low_col]
                        
                        # Update lowest price
                        if current_low < lowest_price:
                            lowest_price = current_low
                            
                            # Check if we should activate trailing stop
                            profit_pct = (entry_price - lowest_price) / entry_price * 100
                            
                            if not trailing_activated and profit_pct >= activation_threshold_pct:
                                trailing_activated = True
                                # Set initial trailing stop position
                                stop_price = lowest_price * (1 + trail_stop_pct/100)
                            elif trailing_activated:
                                # Update trailing stop
                                new_stop = lowest_price * (1 + trail_stop_pct/100)
                                stop_price = min(stop_price, new_stop)
                        
                        # Check exits
                        if initial_stop_pct and current_high >= stop_price:
                            exit_price = stop_price
                            if trailing_activated:
                                exit_type = 'trailing_stop'
                            else:
                                exit_type = 'stop'
                            exit_time = bar['timestamp']
                            break
                        elif target_pct and current_low <= target_price:
                            exit_price = target_price
                            exit_type = 'target'
                            exit_time = bar['timestamp']
                            break
                
                # Calculate return
                if entry['direction'] > 0:
                    raw_return = (exit_price - entry_price) / entry_price
                    max_profit_pct = (highest_price - entry_price) / entry_price * 100
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                    max_profit_pct = (entry_price - lowest_price) / entry_price * 100
                
                net_return = raw_return - (execution_cost_bps * 2 / 10000)
                
                trades.append({
                    'entry_time': entry['time'],
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': entry['direction'],
                    'net_return': net_return,
                    'exit_type': exit_type,
                    'duration_min': (exit_time - entry['time']).total_seconds() / 60,
                    'max_profit_pct': max_profit_pct,
                    'trailing_activated': trailing_activated if 'trailing_activated' in locals() else False
                })
            
            # Update position
            current_pos = signal
            if signal != 0:
                entry = {'time': row['ts'], 'price': row['px'], 'direction': signal}
            else:
                entry = None
    
    return pd.DataFrame(trades) if trades else pd.DataFrame()

# Find best strategy file
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

for idx, (initial_stop, trail_stop, threshold, target, label) in enumerate(THRESHOLD_CONFIGS):
    print(f"\rTesting {idx+1}/{len(THRESHOLD_CONFIGS)}: {label}...", end='', flush=True)
    
    trades_df = extract_trades_with_threshold(
        best_strategy_file,
        market_data,
        initial_stop,
        trail_stop,
        threshold,
        target or 0,
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
        
        # Trailing activation rate
        if 'trailing_activated' in trades_df.columns:
            activation_rate = trades_df['trailing_activated'].mean()
        else:
            activation_rate = 0
        
        results.append({
            'config': label,
            'initial_stop': initial_stop,
            'trail_stop': trail_stop,
            'threshold': threshold,
            'target': target or 0,
            'num_trades': len(trades_df),
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'avg_return_pct': avg_return * 100,
            'stop_exits': exit_counts.get('stop', 0),
            'trailing_stop_exits': exit_counts.get('trailing_stop', 0),
            'target_exits': exit_counts.get('target', 0),
            'signal_exits': exit_counts.get('signal', 0),
            'avg_duration_min': trades_df['duration_min'].mean(),
            'avg_max_profit_pct': trades_df['max_profit_pct'].mean(),
            'trailing_activation_rate': activation_rate
        })

print("\n\nAnalysis complete!")

if results:
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    print("\n" + "=" * 80)
    print("🏆 TOP CONFIGURATIONS BY SHARPE RATIO")
    print("=" * 80)
    
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"\n{row['config']}:")
        print(f"  Sharpe: {row['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {row['win_rate']*100:.1f}%")
        print(f"  Return: {row['total_return']*100:.2f}%")
        print(f"  Avg/Trade: {row['avg_return_pct']:.3f}%")
        print(f"  Trailing Activated: {row['trailing_activation_rate']*100:.1f}% of trades")
        print(f"  Exits: Stop={row['stop_exits']}, Trail={row['trailing_stop_exits']}, "
              f"Target={row['target_exits']}, Signal={row['signal_exits']}")
    
    # Find configs with 90%+ win rate
    high_win_configs = results_df[results_df['win_rate'] >= 0.90]
    
    if len(high_win_configs) > 0:
        print("\n" + "=" * 80)
        print("🎯 CONFIGURATIONS WITH 90%+ WIN RATE")
        print("=" * 80)
        
        for idx, row in high_win_configs.iterrows():
            print(f"\n{row['config']}:")
            print(f"  Win Rate: {row['win_rate']*100:.1f}% ✓")
            print(f"  Sharpe: {row['sharpe_ratio']:.2f}")
            print(f"  Return: {row['total_return']*100:.2f}%")
            print(f"  Trailing Activation: {row['trailing_activation_rate']*100:.1f}% of trades")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Impact of activation threshold
    ax1 = axes[0, 0]
    threshold_impact = results_df.groupby('threshold').agg({
        'sharpe_ratio': 'mean',
        'win_rate': 'mean',
        'trailing_activation_rate': 'mean'
    })
    
    ax1_twin = ax1.twinx()
    ax1.plot(threshold_impact.index, threshold_impact['sharpe_ratio'], 'b-o', label='Sharpe')
    ax1_twin.plot(threshold_impact.index, threshold_impact['win_rate']*100, 'r-s', label='Win Rate %')
    ax1.set_xlabel('Activation Threshold %')
    ax1.set_ylabel('Sharpe Ratio', color='b')
    ax1_twin.set_ylabel('Win Rate %', color='r')
    ax1.set_title('Impact of Activation Threshold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Win Rate vs Sharpe with threshold coloring
    ax2 = axes[0, 1]
    scatter = ax2.scatter(results_df['win_rate']*100, results_df['sharpe_ratio'], 
                         c=results_df['threshold'], cmap='viridis', s=100, alpha=0.7)
    ax2.axvline(90, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Win Rate %')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Performance by Activation Threshold')
    plt.colorbar(scatter, ax=ax2, label='Threshold %')
    
    # 3. Exit type distribution
    ax3 = axes[0, 2]
    # Compare original vs best threshold config
    original = results_df[results_df['threshold'] == 0].iloc[0] if len(results_df[results_df['threshold'] == 0]) > 0 else None
    best_threshold = results_df[results_df['threshold'] > 0].iloc[0] if len(results_df[results_df['threshold'] > 0]) > 0 else None
    
    if original is not None and best_threshold is not None:
        exit_types = ['Stop', 'Trail Stop', 'Target', 'Signal']
        original_exits = [original['stop_exits'], original['trailing_stop_exits'], 
                         original['target_exits'], original['signal_exits']]
        best_exits = [best_threshold['stop_exits'], best_threshold['trailing_stop_exits'],
                     best_threshold['target_exits'], best_threshold['signal_exits']]
        
        x = np.arange(len(exit_types))
        width = 0.35
        
        ax3.bar(x - width/2, original_exits, width, label='No Threshold', alpha=0.7)
        ax3.bar(x + width/2, best_exits, width, label=f"{best_threshold['threshold']}% Threshold", alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(exit_types)
        ax3.set_ylabel('Number of Exits')
        ax3.set_title('Exit Types: Original vs Best Threshold')
        ax3.legend()
    
    # 4. Activation rate by threshold
    ax4 = axes[1, 0]
    activation_by_threshold = results_df.groupby('threshold')['trailing_activation_rate'].mean() * 100
    ax4.bar(activation_by_threshold.index, activation_by_threshold.values, alpha=0.7)
    ax4.set_xlabel('Activation Threshold %')
    ax4.set_ylabel('% of Trades with Trailing Activated')
    ax4.set_title('Trailing Stop Activation Rate')
    ax4.grid(True, alpha=0.3)
    
    # 5. Returns by threshold
    ax5 = axes[1, 1]
    returns_by_threshold = results_df.groupby('threshold')['total_return'].mean() * 100
    avg_return_by_threshold = results_df.groupby('threshold')['avg_return_pct'].mean()
    
    ax5_twin = ax5.twinx()
    ax5.bar(returns_by_threshold.index, returns_by_threshold.values, alpha=0.6, label='Total Return %')
    ax5_twin.plot(avg_return_by_threshold.index, avg_return_by_threshold.values, 'r-o', label='Avg/Trade %')
    ax5.set_xlabel('Activation Threshold %')
    ax5.set_ylabel('Total Return %', color='b')
    ax5_twin.set_ylabel('Avg Return per Trade %', color='r')
    ax5.set_title('Returns by Activation Threshold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Heatmap of win rates
    ax6 = axes[1, 2]
    # Create pivot table for initial stop vs threshold
    pivot_data = results_df.pivot_table(
        values='win_rate',
        index='initial_stop',
        columns='threshold',
        aggfunc='mean'
    ) * 100
    
    if not pivot_data.empty:
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                    center=85, ax=ax6, cbar_kws={'label': 'Win Rate %'})
        ax6.set_title('Win Rate: Initial Stop vs Activation Threshold')
        ax6.set_xlabel('Activation Threshold %')
        ax6.set_ylabel('Initial Stop %')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis of trades that hit the activation threshold
    print("\n" + "=" * 80)
    print("📊 ACTIVATION THRESHOLD ANALYSIS")
    print("=" * 80)
    
    threshold_configs = results_df[results_df['threshold'] > 0].sort_values('threshold')
    
    print("\nActivation rates by threshold:")
    for idx, row in threshold_configs.iterrows():
        print(f"  {row['threshold']:.2f}% threshold → {row['trailing_activation_rate']*100:.1f}% of trades activated trailing")
    
    # Compare performance metrics
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("=" * 80)
    
    original_config = results_df[results_df['threshold'] == 0].iloc[0] if len(results_df[results_df['threshold'] == 0]) > 0 else None
    best_config = results_df.iloc[0]
    
    if original_config is not None:
        print(f"\n1. ORIGINAL vs BEST THRESHOLD:")
        print(f"   Original (no threshold): Sharpe {original_config['sharpe_ratio']:.2f}, Win Rate {original_config['win_rate']*100:.1f}%")
        print(f"   Best ({best_config['config']}): Sharpe {best_config['sharpe_ratio']:.2f}, Win Rate {best_config['win_rate']*100:.1f}%")
        
        if best_config['sharpe_ratio'] > original_config['sharpe_ratio']:
            improvement = (best_config['sharpe_ratio'] / original_config['sharpe_ratio'] - 1) * 100
            print(f"   → {improvement:.1f}% Sharpe improvement!")
    
    print("\n2. OPTIMAL THRESHOLD RANGE:")
    top_5_thresholds = results_df.head(5)['threshold'].values
    print(f"   Best performing thresholds: {np.unique(top_5_thresholds)}")
    
    print("\n3. TRADE-OFFS:")
    print("   - Lower thresholds (0.02-0.05%) activate frequently, similar to no threshold")
    print("   - Medium thresholds (0.05-0.08%) balance protection and breathing room")
    print("   - Higher thresholds (0.08%+) give maximum breathing room but may not activate")
    
    # Save results
    output_path = RESULTS_DIR / 'trailing_stop_threshold_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

else:
    print("\n❌ No valid results generated!")

print("\n" + "=" * 80)