# Comprehensive Bollinger Analysis - Fully Standalone
# This version loads its own market data

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
DATA_DIR = Path('/Users/daws/ADMF-PC/data')

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

print("📊 Comprehensive Bollinger Analysis (Standalone)")
print("=" * 80)

# Load market data
print("Loading market data...")
market_data = None

# Try to find 1-minute data file
for pattern in ['*1m*.csv', '*1min*.csv', '*_1m.csv', 'SPY*.csv']:
    files = list(DATA_DIR.glob(pattern))
    if files:
        for f in files:
            if '1m' in f.name.lower() or '1min' in f.name.lower():
                print(f"Loading {f.name}...")
                market_data = pd.read_csv(f)
                break
        if market_data is not None:
            break

if market_data is None:
    print("❌ Could not find 1-minute market data file!")
    print("Available files in data directory:")
    for f in DATA_DIR.glob('*.csv'):
        print(f"  {f.name}")
else:
    # Clean market data
    print(f"Loaded {len(market_data)} rows of market data")
    
    # Convert timestamp
    if 'timestamp' in market_data.columns:
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
        if market_data['timestamp'].dt.tz is not None:
            market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(None)
    elif 'Datetime' in market_data.columns:
        market_data['timestamp'] = pd.to_datetime(market_data['Datetime'])
        if market_data['timestamp'].dt.tz is not None:
            market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(None)
    
    print(f"Date range: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
    
    # Load strategy index to get parameters
    strategy_index = pd.read_parquet(RESULTS_DIR / 'strategy_index.parquet')
    
    # Get all signal files
    signal_files = list(SIGNAL_DIR.glob('*.parquet'))
    print(f"\nFound {len(signal_files)} signal files")
    
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
                    # Find market data between entry and current signal
                    mask = (market_data['timestamp'] >= entry['time']) & (market_data['timestamp'] <= row['ts'])
                    trade_bars = market_data[mask]
                    
                    exit_price = row['px']
                    exit_type = 'signal'
                    exit_time = row['ts']
                    
                    if len(trade_bars) > 0 and (stop_pct > 0 or target_pct > 0):
                        # Check for stop/target hits
                        entry_price = entry['price']
                        
                        if entry['direction'] > 0:  # Long
                            stop_price = entry_price * (1 - stop_pct/100) if stop_pct > 0 else 0
                            target_price = entry_price * (1 + target_pct/100) if target_pct > 0 else float('inf')
                            
                            for bar_idx, bar in trade_bars.iterrows():
                                if stop_pct > 0 and bar[low_col] <= stop_price:
                                    exit_price = stop_price
                                    exit_type = 'stop'
                                    exit_time = bar['timestamp']
                                    break
                                elif target_pct > 0 and bar[high_col] >= target_price:
                                    exit_price = target_price
                                    exit_type = 'target'
                                    exit_time = bar['timestamp']
                                    break
                        else:  # Short
                            stop_price = entry_price * (1 + stop_pct/100) if stop_pct > 0 else float('inf')
                            target_price = entry_price * (1 - target_pct/100) if target_pct > 0 else 0
                            
                            for bar_idx, bar in trade_bars.iterrows():
                                if stop_pct > 0 and bar[high_col] >= stop_price:
                                    exit_price = stop_price
                                    exit_type = 'stop'
                                    exit_time = bar['timestamp']
                                    break
                                elif target_pct > 0 and bar[low_col] <= target_price:
                                    exit_price = target_price
                                    exit_type = 'target'
                                    exit_time = bar['timestamp']
                                    break
                    
                    # Calculate return
                    if entry['direction'] > 0:
                        raw_return = (exit_price - entry['price']) / entry['price']
                    else:
                        raw_return = (entry['price'] - exit_price) / entry['price']
                    
                    net_return = raw_return - (execution_cost_bps * 2 / 10000)
                    
                    trades.append({
                        'entry_time': entry['time'],
                        'exit_time': exit_time,
                        'entry_price': entry['price'],
                        'exit_price': exit_price,
                        'direction': entry['direction'],
                        'raw_return': raw_return,
                        'net_return': net_return,
                        'exit_type': exit_type,
                        'duration_min': (exit_time - entry['time']).total_seconds() / 60
                    })
                
                # Update position
                current_pos = signal
                if signal != 0:
                    entry = {'time': row['ts'], 'price': row['px'], 'direction': signal, 'signal_idx': _}
                else:
                    entry = None
        
        return pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Find strategies with good signal counts
    print("\nFinding strategies with sufficient signals...")
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
    
    if len(good_strategies) == 0:
        print("❌ No strategies with sufficient signals found!")
    else:
        # Test top strategies
        results = []
        strategies_to_test = good_strategies[:10]  # Test top 10
        
        print(f"\nTesting {len(strategies_to_test)} strategies with {len(STOP_TARGET_CONFIGS)} stop/target configs...")
        
        for i, strategy in enumerate(strategies_to_test):
            print(f"\rProcessing strategy {i+1}/{len(strategies_to_test)}...", end='', flush=True)
            
            for stop_pct, target_pct in STOP_TARGET_CONFIGS:
                try:
                    trades_df = extract_and_test_stops(
                        strategy['file'], 
                        market_data, 
                        stop_pct, 
                        target_pct,
                        execution_cost_bps=1.0
                    )
                    
                    if len(trades_df) > 10:  # Need minimum trades
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
                            if trading_days == 0:
                                trading_days = 1
                            trades_per_day = len(trades_df) / trading_days
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
                except Exception as e:
                    print(f"\nError processing {strategy['file'].name}: {e}")
                    continue
        
        print("\n\nProcessing complete!")
        
        if results:
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            print(f"\nGenerated {len(results_df)} result rows")
            
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
            
            # Aggregate by stop/target combination
            print("\n" + "=" * 80)
            print("📊 AVERAGE PERFORMANCE BY STOP/TARGET")
            print("=" * 80)
            
            agg_df = results_df.groupby(['stop_pct', 'target_pct']).agg({
                'sharpe_ratio': 'mean',
                'total_return': 'mean',
                'win_rate': 'mean',
                'num_trades': 'mean'
            }).round(3)
            
            for (stop, target), row in agg_df.iterrows():
                print(f"Stop: {stop}%, Target: {target}% → "
                      f"Avg Sharpe: {row['sharpe_ratio']:.2f}, "
                      f"Avg Return: {row['total_return']*100:.2f}%, "
                      f"Win Rate: {row['win_rate']*100:.1f}%")
            
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
            if not pivot_sharpe.empty:
                sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
                ax.set_title('Average Sharpe Ratio by Stop/Target %')
            
            # 2. Scatter plot of Sharpe vs Return
            ax = axes[0, 1]
            ax.scatter(results_df['total_return'] * 100, results_df['sharpe_ratio'], alpha=0.6)
            ax.set_xlabel('Total Return %')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Sharpe vs Return for All Configurations')
            ax.grid(True, alpha=0.3)
            
            # 3. Win rate by configuration
            ax = axes[1, 0]
            config_labels = [f"{row['stop_pct']}/{row['target_pct']}" 
                           for _, row in results_df.iterrows()]
            unique_configs = results_df.groupby(['stop_pct', 'target_pct'])['win_rate'].mean() * 100
            ax.bar(range(len(unique_configs)), unique_configs.values)
            ax.set_xticks(range(len(unique_configs)))
            ax.set_xticklabels([f"{s}/{t}" for s, t in unique_configs.index], rotation=45)
            ax.set_ylabel('Average Win Rate %')
            ax.set_title('Win Rate by Stop/Target Configuration')
            
            # 4. Summary stats
            ax = axes[1, 1]
            ax.axis('off')
            summary_text = f"Analysis Summary\n\n"
            summary_text += f"Strategies tested: {len(strategies_to_test)}\n"
            summary_text += f"Configurations tested: {len(STOP_TARGET_CONFIGS)}\n"
            summary_text += f"Total results: {len(results_df)}\n\n"
            
            # Best configuration
            best_row = results_df.nlargest(1, 'sharpe_ratio').iloc[0]
            summary_text += f"Best Configuration:\n"
            summary_text += f"  Strategy: {best_row['strategy']}\n"
            summary_text += f"  Stop/Target: {best_row['stop_pct']}%/{best_row['target_pct']}%\n"
            summary_text += f"  Sharpe: {best_row['sharpe_ratio']:.2f}\n"
            summary_text += f"  Return: {best_row['total_return']*100:.2f}%\n"
            summary_text += f"  Win Rate: {best_row['win_rate']*100:.1f}%"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=12, family='monospace')
            
            plt.tight_layout()
            plt.show()
            
            # Save results
            output_path = RESULTS_DIR / 'stop_target_analysis_standalone.csv'
            results_df.to_csv(output_path, index=False)
            print(f"\n✅ Results saved to: {output_path}")
            
        else:
            print("\n❌ No valid results generated!")

print("\n💡 Analysis complete!")