# Analyze Bollinger Signals Directly from Parquet Files
# This bypasses the performance_df which shows 0 trades

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
# Use absolute path or check if run_dir exists in globals
if 'run_dir' in globals():
    run_dir = Path(run_dir)
else:
    # Use the specific directory from your run
    run_dir = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250625_185742')
    
execution_cost_bps = 1.0
SAMPLE_SIZE = 10  # Number of strategies to analyze in detail

print(f"Using run_dir: {run_dir}")
print(f"Run dir exists: {run_dir.exists()}")

print("📊 Direct Bollinger Signal Analysis")
print("=" * 80)

# Load market data if not already loaded
if 'market_data' not in globals():
    print("Loading market data...")
    # Try to find market data
    data_files = list(Path('data').glob('*1m*.csv'))
    if data_files:
        market_data = pd.read_csv(data_files[0])
        market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
        if market_data['timestamp'].dt.tz is not None:
            market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(None)
        print(f"Loaded market data from {data_files[0]}")
    else:
        print("❌ Could not find market data file")
else:
    # Clean timestamps
    if market_data['timestamp'].dt.tz is not None:
        market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(None)

# Find all signal files
signal_dir = run_dir / 'traces' / 'signals' / 'bollinger_bands'
parquet_files = list(signal_dir.glob('*.parquet'))
print(f"\nFound {len(parquet_files)} signal files")

# Function to extract strategy parameters from filename
def parse_strategy_params(filename):
    """Extract period and std_dev from filename like SPY_1m_strategy_81.parquet"""
    # Load the strategy index to get parameters
    strategy_index_path = run_dir / 'strategy_index.parquet'
    if strategy_index_path.exists():
        index_df = pd.read_parquet(strategy_index_path)
        # Extract strategy number from filename
        import re
        match = re.search(r'strategy_(\d+)\.parquet', filename)
        if match:
            strategy_num = int(match.group(1))
            if strategy_num < len(index_df):
                row = index_df.iloc[strategy_num]
                return row.get('period', 'N/A'), row.get('std_dev', 'N/A')
    return 'N/A', 'N/A'

# Analyze each file
all_results = []

for i, parquet_file in enumerate(parquet_files[:SAMPLE_SIZE]):
    print(f"\nAnalyzing {parquet_file.name}...")
    
    # Load signals
    df = pd.read_parquet(parquet_file)
    df['ts'] = pd.to_datetime(df['ts'])
    if hasattr(df['ts'].dtype, 'tz'):
        df['ts'] = df['ts'].dt.tz_localize(None)
    
    # Get strategy parameters
    period, std_dev = parse_strategy_params(parquet_file.name)
    
    # Count signals
    signal_counts = df['val'].value_counts()
    non_zero_signals = (df['val'] != 0).sum()
    
    if non_zero_signals == 0:
        print(f"  No signals found")
        continue
    
    # Extract trades
    df = df.sort_values('ts')
    trades = []
    current_position = 0
    entry_time = None
    entry_price = None
    
    for idx, row in df.iterrows():
        signal = row['val']
        
        if current_position == 0 and signal != 0:
            # Entry
            current_position = signal
            entry_time = row['ts']
            entry_price = row['px']
        elif current_position != 0 and signal != current_position:
            # Exit
            if entry_time is not None:
                exit_time = row['ts']
                exit_price = row['px']
                
                # Calculate return
                if current_position > 0:  # Long
                    raw_return = (exit_price - entry_price) / entry_price
                else:  # Short
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - (execution_cost_bps * 2 / 10000)
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': current_position,
                    'raw_return': raw_return,
                    'net_return': net_return,
                    'duration_minutes': (exit_time - entry_time).total_seconds() / 60
                })
            
            # Update position
            current_position = signal
            if signal != 0:
                entry_time = row['ts']
                entry_price = row['px']
            else:
                entry_time = None
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        total_return = (1 + trades_df['net_return']).prod() - 1
        win_rate = (trades_df['net_return'] > 0).mean()
        avg_return = trades_df['net_return'].mean()
        
        # Sharpe ratio (approximate)
        if trades_df['net_return'].std() > 0:
            trading_days = (df['ts'].max() - df['ts'].min()).days
            trades_per_day = len(trades_df) / max(trading_days, 1)
            sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * trades_per_day)
        else:
            sharpe = 0
        
        result = {
            'file': parquet_file.name,
            'period': period,
            'std_dev': std_dev,
            'num_signals': len(df),
            'non_zero_signals': non_zero_signals,
            'num_trades': len(trades_df),
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'sharpe_ratio': sharpe,
            'avg_duration_minutes': trades_df['duration_minutes'].mean(),
            'long_trades': (trades_df['direction'] > 0).sum(),
            'short_trades': (trades_df['direction'] < 0).sum()
        }
        
        all_results.append(result)
        
        print(f"  Signals: {non_zero_signals} | Trades: {len(trades_df)}")
        print(f"  Return: {total_return*100:.2f}% | Win Rate: {win_rate*100:.1f}%")
        print(f"  Sharpe: {sharpe:.2f} | Avg Duration: {trades_df['duration_minutes'].mean():.1f} min")

# Create summary DataFrame
if all_results:
    summary_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nAnalyzed {len(summary_df)} strategies with trades")
    print(f"Average trades per strategy: {summary_df['num_trades'].mean():.1f}")
    print(f"Average return: {summary_df['total_return'].mean()*100:.2f}%")
    print(f"Average Sharpe: {summary_df['sharpe_ratio'].mean():.2f}")
    print(f"Average win rate: {summary_df['win_rate'].mean()*100:.1f}%")
    
    # Top performers
    print("\nTop 5 by Sharpe Ratio:")
    top_sharpe = summary_df.nlargest(5, 'sharpe_ratio')
    for idx, row in top_sharpe.iterrows():
        print(f"  {row['file']}: Sharpe={row['sharpe_ratio']:.2f}, Return={row['total_return']*100:.2f}%")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Sharpe distribution
    ax = axes[0, 0]
    summary_df['sharpe_ratio'].hist(bins=20, ax=ax)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Sharpe Ratio Distribution')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Return distribution
    ax = axes[0, 1]
    (summary_df['total_return'] * 100).hist(bins=20, ax=ax)
    ax.set_xlabel('Total Return %')
    ax.set_ylabel('Count')
    ax.set_title('Return Distribution')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 3. Win rate vs Sharpe
    ax = axes[1, 0]
    ax.scatter(summary_df['win_rate'] * 100, summary_df['sharpe_ratio'], alpha=0.6)
    ax.set_xlabel('Win Rate %')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Win Rate vs Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    
    # 4. Trade frequency vs return
    ax = axes[1, 1]
    ax.scatter(summary_df['num_trades'], summary_df['total_return'] * 100, alpha=0.6)
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Total Return %')
    ax.set_title('Trade Frequency vs Return')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    output_path = run_dir / 'direct_signal_analysis.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Now test stop/target on best strategy
    best_strategy = summary_df.nlargest(1, 'sharpe_ratio').iloc[0]
    print(f"\n🎯 Testing stop/target on best strategy: {best_strategy['file']}")
    print(f"   Base Sharpe: {best_strategy['sharpe_ratio']:.2f}")
    print(f"   Base Return: {best_strategy['total_return']*100:.2f}%")
    
    # Reload and test with stops
    df = pd.read_parquet(signal_dir / best_strategy['file'])
    df['ts'] = pd.to_datetime(df['ts'])
    if hasattr(df['ts'].dtype, 'tz'):
        df['ts'] = df['ts'].dt.tz_localize(None)
    
    # Test different stop/target combinations
    stop_target_results = []
    
    for stop_pct in [0.03, 0.05, 0.075, 0.1]:
        for target_pct in [0.05, 0.075, 0.1, 0.15]:
            if target_pct > stop_pct:  # Only test reasonable combinations
                # Run same trade extraction but with stop/target logic
                # (simplified version for demonstration)
                print(f"   Testing Stop={stop_pct}%, Target={target_pct}%...")
                # This would need the full stop/target implementation
                
else:
    print("\n❌ No strategies produced valid trades")

print("\n" + "=" * 80)
print("💡 Direct analysis complete!")
print("=" * 80)