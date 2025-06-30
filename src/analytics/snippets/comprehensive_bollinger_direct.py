# Comprehensive Bollinger Analysis - Direct from Signals
# This bypasses the faulty performance_df and analyzes signals directly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250625_185742')
SIGNAL_DIR = RESULTS_DIR / 'traces/signals/bollinger_bands'
STOP_TARGET_CONFIGS = [
    (0.03, 0.05),    # Ultra-tight for 1m data
    (0.05, 0.075),   # Very tight
    (0.05, 0.10),    # 2:1 reward/risk
    (0.075, 0.10),   # Proven optimal on 5m
    (0.075, 0.15),   # 2:1 
    (0.10, 0.15),    # Wider
    (0.10, 0.20),    # 2:1 wider
    (0, 0),          # Baseline
]

print("📊 Comprehensive Bollinger Analysis (Direct from Signals)")
print("=" * 80)

# Load market data
if 'market_data' not in globals():
    print("❌ Please load market_data first!")
else:
    # Clean timestamps
    if market_data['timestamp'].dt.tz is not None:
        market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(None)
    
    # Load strategy index to get parameters
    strategy_index = pd.read_parquet(RESULTS_DIR / 'strategy_index.parquet')
    
    # Get all signal files
    signal_files = list(SIGNAL_DIR.glob('*.parquet'))
    print(f"Found {len(signal_files)} signal files")
    
    # Function to extract trades with stop/target
    def extract_and_test_stops(signal_file, market_data, stop_pct, target_pct, execution_cost_bps=1.0):
        """Extract trades and apply stop/target logic"""
        # Load signals
        df = pd.read_parquet(signal_file)
        df['ts'] = pd.to_datetime(df['ts'])
        if hasattr(df['ts'].dtype, 'tz'):
            df['ts'] = df['ts'].dt.tz_localize(None)
        
        df = df.sort_values('ts')
        
        # Extract trades with stop/target logic
        trades = []
        current_pos = 0
        entry = None
        
        # Determine column names
        close_col = 'Close' if 'Close' in market_data.columns else 'close'
        low_col = 'Low' if 'Low' in market_data.columns else 'low'
        high_col = 'High' if 'High' in market_data.columns else 'high'
        
        for _, row in df.iterrows():
            signal = row['val']
            
            if current_pos == 0 and signal != 0:
                # Entry
                current_pos = signal
                entry = {
                    'time': row['ts'], 
                    'price': row['px'], 
                    'direction': signal,
                    'signal_idx': _
                }
                
            elif current_pos != 0 and signal != current_pos:
                # Exit - but check for stop/target first
                if entry:
                    # Find market data between entry and exit
                    mask = (market_data['timestamp'] >= entry['time']) & (market_data['timestamp'] <= row['ts'])
                    trade_bars = market_data[mask]
                    
                    exit_price = row['px']
                    exit_type = 'signal'
                    
                    if len(trade_bars) > 0 and (stop_pct > 0 or target_pct > 0):
                        # Check for stop/target hits
                        entry_price = entry['price']
                        
                        if entry['direction'] > 0:  # Long
                            stop_price = entry_price * (1 - stop_pct/100) if stop_pct > 0 else 0
                            target_price = entry_price * (1 + target_pct/100) if target_pct > 0 else float('inf')
                            
                            for _, bar in trade_bars.iterrows():
                                if stop_pct > 0 and bar[low_col] <= stop_price:
                                    exit_price = stop_price
                                    exit_type = 'stop'
                                    break
                                elif target_pct > 0 and bar[high_col] >= target_price:
                                    exit_price = target_price
                                    exit_type = 'target'
                                    break
                        else:  # Short
                            stop_price = entry_price * (1 + stop_pct/100) if stop_pct > 0 else float('inf')
                            target_price = entry_price * (1 - target_pct/100) if target_pct > 0 else 0
                            
                            for _, bar in trade_bars.iterrows():
                                if stop_pct > 0 and bar[high_col] >= stop_price:
                                    exit_price = stop_price
                                    exit_type = 'stop'
                                    break
                                elif target_pct > 0 and bar[low_col] <= target_price:
                                    exit_price = target_price
                                    exit_type = 'target'
                                    break
                    
                    # Calculate return
                    if entry['direction'] > 0:
                        raw_return = (exit_price - entry['price']) / entry['price']
                    else:
                        raw_return = (entry['price'] - exit_price) / entry['price']
                    
                    net_return = raw_return - (execution_cost_bps * 2 / 10000)
                    
                    trades.append({
                        'entry_time': entry['time'],
                        'exit_time': row['ts'] if exit_type == 'signal' else entry['time'],
                        'entry_price': entry['price'],
                        'exit_price': exit_price,
                        'direction': entry['direction'],
                        'raw_return': raw_return,
                        'net_return': net_return,
                        'exit_type': exit_type,
                        'duration_min': (row['ts'] - entry['time']).total_seconds() / 60
                    })
                
                # Update position
                current_pos = signal
                if signal != 0:
                    entry = {'time': row['ts'], 'price': row['px'], 'direction': signal, 'signal_idx': _}
                else:
                    entry = None
        
        return pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Analyze a sample of strategies
    print("\nAnalyzing strategies with different stop/target configurations...")
    
    # Find strategies with good signal counts
    good_strategies = []
    for i, signal_file in enumerate(signal_files):
        df = pd.read_parquet(signal_file)
        non_zero = (df['val'] != 0).sum()
        if non_zero > 500:  # Strategies with decent signal count
            # Get strategy parameters
            strategy_num = int(signal_file.stem.split('_')[-1])
            if strategy_num < len(strategy_index):
                params = strategy_index.iloc[strategy_num]
                good_strategies.append({
                    'file': signal_file,
                    'strategy_num': strategy_num,
                    'period': params.get('period', 'N/A'),
                    'std_dev': params.get('std_dev', 'N/A'),
                    'signal_count': non_zero
                })
    
    print(f"Found {len(good_strategies)} strategies with >500 signals")
    
    # Test top 10 strategies
    results = []
    strategies_to_test = good_strategies[:10]
    
    for i, strategy in enumerate(strategies_to_test):
        print(f"\rTesting strategy {i+1}/{len(strategies_to_test)}...", end='', flush=True)
        
        for stop_pct, target_pct in STOP_TARGET_CONFIGS:
            trades_df = extract_and_test_stops(
                strategy['file'], 
                market_data, 
                stop_pct, 
                target_pct,
                execution_cost_bps=1.0
            )
            
            if len(trades_df) > 0:
                # Calculate metrics
                total_return = (1 + trades_df['net_return']).prod() - 1
                win_rate = (trades_df['net_return'] > 0).mean()
                avg_return = trades_df['net_return'].mean()
                
                # Exit type counts
                exit_counts = trades_df['exit_type'].value_counts()
                stop_pct_count = exit_counts.get('stop', 0) / len(trades_df) * 100
                target_pct_count = exit_counts.get('target', 0) / len(trades_df) * 100
                
                # Sharpe ratio
                if trades_df['net_return'].std() > 0:
                    trading_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
                    trades_per_day = len(trades_df) / max(trading_days, 1)
                    sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * trades_per_day)
                else:
                    sharpe = 0
                
                results.append({
                    'strategy': f"P{strategy['period']}_S{strategy['std_dev']}",
                    'stop_pct': stop_pct,
                    'target_pct': target_pct,
                    'num_trades': len(trades_df),
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'stop_hits': stop_pct_count,
                    'target_hits': target_pct_count,
                    'avg_duration': trades_df['duration_min'].mean()
                })
    
    print("\n\nAnalysis complete!")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best configurations
    print("\n" + "=" * 80)
    print("🎯 OPTIMAL STOP/TARGET CONFIGURATIONS")
    print("=" * 80)
    
    # Best overall by Sharpe
    best_sharpe = results_df.nlargest(10, 'sharpe_ratio')
    print("\nTop 10 by Sharpe Ratio:")
    for idx, row in best_sharpe.iterrows():
        print(f"{row['strategy']} | Stop: {row['stop_pct']}%, Target: {row['target_pct']}% → "
              f"Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']*100:.2f}%, "
              f"Trades: {row['num_trades']}")
    
    # Best by return
    best_return = results_df.nlargest(10, 'total_return')
    print("\nTop 10 by Total Return:")
    for idx, row in best_return.iterrows():
        print(f"{row['strategy']} | Stop: {row['stop_pct']}%, Target: {row['target_pct']}% → "
              f"Return: {row['total_return']*100:.2f}%, Sharpe: {row['sharpe_ratio']:.2f}, "
              f"Win Rate: {row['win_rate']*100:.1f}%")
    
    # Aggregate by stop/target combination
    agg_results = results_df.groupby(['stop_pct', 'target_pct']).agg({
        'sharpe_ratio': ['mean', 'std'],
        'total_return': ['mean', 'std'],
        'win_rate': 'mean',
        'stop_hits': 'mean',
        'target_hits': 'mean'
    }).round(3)
    
    print("\n" + "=" * 80)
    print("📊 AGGREGATE PERFORMANCE BY STOP/TARGET")
    print("=" * 80)
    print(agg_results)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Heatmap of average Sharpe by stop/target
    ax = axes[0, 0]
    pivot_sharpe = results_df.pivot_table(
        values='sharpe_ratio',
        index='stop_pct',
        columns='target_pct',
        aggfunc='mean'
    )
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Average Sharpe Ratio by Stop/Target %')
    
    # 2. Heatmap of average return
    ax = axes[0, 1]
    pivot_return = results_df.pivot_table(
        values='total_return',
        index='stop_pct',
        columns='target_pct',
        aggfunc='mean'
    ) * 100
    sns.heatmap(pivot_return, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Average Total Return % by Stop/Target')
    
    # 3. Win rate distribution
    ax = axes[1, 0]
    for stop in [0.05, 0.075, 0.1]:
        data = results_df[results_df['stop_pct'] == stop]
        ax.plot(data['target_pct'], data['win_rate'] * 100, 
                marker='o', label=f'Stop={stop}%')
    ax.set_xlabel('Target %')
    ax.set_ylabel('Win Rate %')
    ax.set_title('Win Rate by Target (for different stops)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Exit type distribution
    ax = axes[1, 1]
    stop_target_075_10 = results_df[(results_df['stop_pct'] == 0.075) & (results_df['target_pct'] == 0.1)]
    if len(stop_target_075_10) > 0:
        avg_stops = stop_target_075_10['stop_hits'].mean()
        avg_targets = stop_target_075_10['target_hits'].mean()
        avg_signals = 100 - avg_stops - avg_targets
        
        ax.bar(['Stops', 'Targets', 'Signals'], [avg_stops, avg_targets, avg_signals])
        ax.set_ylabel('Percentage of Exits')
        ax.set_title('Exit Type Distribution (0.075% Stop / 0.1% Target)')
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    output_path = RESULTS_DIR / 'stop_target_analysis_direct.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Special focus on 0.075/0.1 configuration
    stop_075_target_10 = results_df[(results_df['stop_pct'] == 0.075) & (results_df['target_pct'] == 0.1)]
    if len(stop_075_target_10) > 0:
        print("\n" + "=" * 80)
        print("🎯 FOCUS: 0.075% Stop / 0.1% Target Performance")
        print("=" * 80)
        print(f"Average Sharpe: {stop_075_target_10['sharpe_ratio'].mean():.2f}")
        print(f"Average Return: {stop_075_target_10['total_return'].mean()*100:.2f}%")
        print(f"Average Win Rate: {stop_075_target_10['win_rate'].mean()*100:.1f}%")
        print(f"Stop hits: {stop_075_target_10['stop_hits'].mean():.1f}%")
        print(f"Target hits: {stop_075_target_10['target_hits'].mean():.1f}%")

print("\n💡 Analysis complete!")