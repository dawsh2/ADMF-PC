# Comprehensive Bollinger Analysis - Full Featured Standalone Version
# Includes all features from comprehensive_1m_bollinger_analysis.py

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

ANALYZE_ALL_STRATEGIES = True  # Set to False to only analyze top N
TOP_N_STRATEGIES = 20
execution_cost_bps = 1.0

STOP_TARGET_CONFIGS = [
    # Tight stops for 1m data
    (0.03, 0.05),    # Ultra-tight
    (0.05, 0.075),   # Very tight
    (0.05, 0.10),    # 2:1 reward/risk
    (0.075, 0.10),   # Proven optimal on 5m
    (0.075, 0.15),   # 2:1 
    (0.10, 0.15),    # Wider
    (0.10, 0.20),    # 2:1 wider
    (0, 0),          # Baseline
]

# Regime-specific configurations
REGIME_CONFIGS = {
    'Low Vol': [(0.03, 0.05), (0.05, 0.075), (0.05, 0.10)],  # Tighter for low vol
    'Medium Vol': [(0.05, 0.10), (0.075, 0.10), (0.075, 0.15)],  # Standard
    'High Vol': [(0.075, 0.15), (0.10, 0.15), (0.10, 0.20)],  # Wider for high vol
    'Trending Up': [(0.05, 0.15), (0.075, 0.20), (0.10, 0.30)],  # Favor upside
    'Trending Down': [(0.10, 0.075), (0.15, 0.10), (0.20, 0.15)],  # Tighter targets
    'Ranging': [(0.05, 0.05), (0.075, 0.075), (0.10, 0.10)]  # Symmetric
}

print("📊 Comprehensive 1-Minute Bollinger Analysis (Full Featured)")
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
    exit()

# Clean market data
print(f"Loaded {len(market_data)} rows of market data")

# Convert timestamp - handle timezone-aware strings
if 'timestamp' in market_data.columns:
    # Parse with UTC=True to handle timezone info, then remove timezone
    market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], utc=True).dt.tz_localize(None)
elif 'Datetime' in market_data.columns:
    market_data['timestamp'] = pd.to_datetime(market_data['Datetime'], utc=True).dt.tz_localize(None)
else:
    print("Warning: No timestamp column found!")
    print(f"Available columns: {list(market_data.columns)}")

# Calculate market regimes
def calculate_market_regimes(market_data):
    """Calculate volatility and trend regimes"""
    # Determine the close column name
    close_col = 'Close' if 'Close' in market_data.columns else 'close'
    
    # Volatility regime (20-period rolling)
    market_data['returns'] = market_data[close_col].pct_change()
    market_data['volatility'] = market_data['returns'].rolling(window=20*60).std() * np.sqrt(252*390)  # 390 1min bars/day
    
    vol_percentiles = market_data['volatility'].quantile([0.33, 0.67])
    market_data['vol_regime'] = pd.cut(
        market_data['volatility'],
        bins=[0, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
        labels=['Low Vol', 'Medium Vol', 'High Vol']
    )
    
    # Trend regime (using 60-period SMA)
    market_data['sma_60'] = market_data[close_col].rolling(window=60).mean()
    market_data['sma_240'] = market_data[close_col].rolling(window=240).mean()
    
    # Trend strength
    market_data['trend_strength'] = (market_data[close_col] - market_data['sma_240']) / market_data['sma_240'] * 100
    
    # Classify trend
    conditions = [
        (market_data['trend_strength'] > 0.5) & (market_data['sma_60'] > market_data['sma_240']),
        (market_data['trend_strength'] < -0.5) & (market_data['sma_60'] < market_data['sma_240']),
        (market_data['trend_strength'].abs() <= 0.5)
    ]
    choices = ['Trending Up', 'Trending Down', 'Ranging']
    market_data['trend_regime'] = np.select(conditions, choices, default='Ranging')
    
    return market_data

print("Calculating market regimes...")
market_data = calculate_market_regimes(market_data)

# Get actual trading period
actual_trading_days = len(market_data['timestamp'].dt.date.unique())
date_range = f"{market_data['timestamp'].min().date()} to {market_data['timestamp'].max().date()}"
print(f"Actual trading period: {date_range} ({actual_trading_days} days)")

# Load strategy index
strategy_index = pd.read_parquet(RESULTS_DIR / 'strategy_index.parquet')

# Get all signal files
signal_files = list(SIGNAL_DIR.glob('*.parquet'))
print(f"\nFound {len(signal_files)} signal files")

# Function to extract trades with stop/target and regime info
def analyze_strategy_with_regimes(signal_file, market_data, strategy_params):
    """Analyze strategy performance across different regimes and stop/target configs"""
    
    # Load signals
    df = pd.read_parquet(signal_file)
    df['ts'] = pd.to_datetime(df['ts'])
    if hasattr(df['ts'].dtype, 'tz'):
        df['ts'] = df['ts'].dt.tz_localize(None)
    
    df = df.sort_values('ts')
    
    # Determine column names
    close_col = 'Close' if 'Close' in market_data.columns else 'close'
    low_col = 'Low' if 'Low' in market_data.columns else 'low'
    high_col = 'High' if 'High' in market_data.columns else 'high'
    
    results = []
    
    # Test each stop/target configuration
    for stop_pct, target_pct in STOP_TARGET_CONFIGS:
        # Extract trades with this stop/target
        trades = []
        current_pos = 0
        entry = None
        
        for idx, row in df.iterrows():
            signal = row['val']
            
            if current_pos == 0 and signal != 0:
                # Entry
                current_pos = signal
                # Find market data index for entry time
                time_diff = (market_data['timestamp'] - row['ts']).abs()
                market_idx = time_diff.idxmin()
                
                entry = {
                    'time': row['ts'], 
                    'price': row['px'], 
                    'direction': signal,
                    'market_idx': market_idx,
                    'vol_regime': market_data.loc[market_idx, 'vol_regime'],
                    'trend_regime': market_data.loc[market_idx, 'trend_regime']
                }
                
            elif current_pos != 0 and signal != current_pos:
                # Exit - check for stop/target first
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
                        'duration_min': (exit_time - entry['time']).total_seconds() / 60,
                        'vol_regime': entry['vol_regime'],
                        'trend_regime': entry['trend_regime']
                    })
                
                # Update position
                current_pos = signal
                if signal != 0:
                    time_diff = (market_data['timestamp'] - row['ts']).abs()
                    market_idx = time_diff.idxmin()
                    entry = {
                        'time': row['ts'], 
                        'price': row['px'], 
                        'direction': signal,
                        'market_idx': market_idx,
                        'vol_regime': market_data.loc[market_idx, 'vol_regime'],
                        'trend_regime': market_data.loc[market_idx, 'trend_regime']
                    }
                else:
                    entry = None
        
        # Convert to DataFrame
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Calculate overall metrics
            total_return = (1 + trades_df['net_return']).prod() - 1
            win_rate = (trades_df['net_return'] > 0).mean()
            
            # Sharpe ratio
            if trades_df['net_return'].std() > 0:
                trades_per_day = len(trades_df) / actual_trading_days
                sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * trades_per_day)
            else:
                sharpe = 0
            
            # Exit type percentages
            exit_counts = trades_df['exit_type'].value_counts()
            stop_hits = exit_counts.get('stop', 0) / len(trades_df) * 100
            target_hits = exit_counts.get('target', 0) / len(trades_df) * 100
            
            # Store overall result
            results.append({
                'stop_pct': stop_pct,
                'target_pct': target_pct,
                'regime': 'Overall',
                'num_trades': len(trades_df),
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'stop_hits': stop_hits,
                'target_hits': target_hits,
                'avg_duration': trades_df['duration_min'].mean(),
                **strategy_params
            })
            
            # Calculate metrics by regime
            for vol_regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                regime_trades = trades_df[trades_df['vol_regime'] == vol_regime]
                if len(regime_trades) > 5:  # Need minimum trades
                    regime_return = (1 + regime_trades['net_return']).prod() - 1
                    regime_win_rate = (regime_trades['net_return'] > 0).mean()
                    
                    if regime_trades['net_return'].std() > 0:
                        regime_sharpe = regime_trades['net_return'].mean() / regime_trades['net_return'].std() * np.sqrt(252 * len(regime_trades) / actual_trading_days)
                    else:
                        regime_sharpe = 0
                    
                    results.append({
                        'stop_pct': stop_pct,
                        'target_pct': target_pct,
                        'regime': vol_regime,
                        'num_trades': len(regime_trades),
                        'total_return': regime_return,
                        'sharpe_ratio': regime_sharpe,
                        'win_rate': regime_win_rate,
                        'stop_hits': 0,  # Could calculate if needed
                        'target_hits': 0,
                        'avg_duration': regime_trades['duration_min'].mean(),
                        **strategy_params
                    })
            
            # Also by trend regime
            for trend_regime in ['Trending Up', 'Trending Down', 'Ranging']:
                regime_trades = trades_df[trades_df['trend_regime'] == trend_regime]
                if len(regime_trades) > 5:
                    regime_return = (1 + regime_trades['net_return']).prod() - 1
                    regime_win_rate = (regime_trades['net_return'] > 0).mean()
                    
                    if regime_trades['net_return'].std() > 0:
                        regime_sharpe = regime_trades['net_return'].mean() / regime_trades['net_return'].std() * np.sqrt(252 * len(regime_trades) / actual_trading_days)
                    else:
                        regime_sharpe = 0
                    
                    results.append({
                        'stop_pct': stop_pct,
                        'target_pct': target_pct,
                        'regime': trend_regime,
                        'num_trades': len(regime_trades),
                        'total_return': regime_return,
                        'sharpe_ratio': regime_sharpe,
                        'win_rate': regime_win_rate,
                        'stop_hits': 0,
                        'target_hits': 0,
                        'avg_duration': regime_trades['duration_min'].mean(),
                        **strategy_params
                    })
    
    return pd.DataFrame(results) if results else None

# Find strategies with sufficient signals
print("\nAnalyzing signal counts...")
strategy_signal_counts = []

for signal_file in signal_files:
    df = pd.read_parquet(signal_file)
    non_zero = (df['val'] != 0).sum()
    
    # Get strategy parameters
    strategy_num = int(signal_file.stem.split('_')[-1])
    if strategy_num < len(strategy_index):
        params = strategy_index.iloc[strategy_num]
        strategy_signal_counts.append({
            'file': signal_file,
            'strategy_num': strategy_num,
            'period': params.get('period', 'N/A'),
            'std_dev': params.get('std_dev', 'N/A'),
            'signal_count': non_zero,
            'strategy_hash': params.get('strategy_hash', '')
        })

# Sort by signal count
strategy_signal_counts = sorted(strategy_signal_counts, key=lambda x: x['signal_count'], reverse=True)

# Determine which strategies to analyze
if ANALYZE_ALL_STRATEGIES:
    strategies_to_analyze = [s for s in strategy_signal_counts if s['signal_count'] > 100]
    print(f"\nAnalyzing ALL {len(strategies_to_analyze)} strategies with >100 signals...")
else:
    strategies_to_analyze = strategy_signal_counts[:TOP_N_STRATEGIES]
    print(f"\nAnalyzing top {len(strategies_to_analyze)} strategies by signal count")

# Show signal distribution
print("\nSignal count distribution:")
bins = [0, 100, 500, 1000, 2000, 5000, 10000]
for i in range(len(bins)-1):
    count = sum(1 for s in strategy_signal_counts if bins[i] < s['signal_count'] <= bins[i+1])
    print(f"  ({bins[i]}, {bins[i+1]}]: {count} strategies")

# Main analysis loop
all_results = []

for idx, strategy in enumerate(strategies_to_analyze):
    print(f"\rProcessing strategy {idx+1}/{len(strategies_to_analyze)}...", end='', flush=True)
    
    try:
        strategy_params = {
            'strategy': f"P{strategy['period']}_S{strategy['std_dev']}",
            'period': strategy['period'],
            'std_dev': strategy['std_dev'],
            'strategy_hash': strategy['strategy_hash'][:8]
        }
        
        results = analyze_strategy_with_regimes(strategy['file'], market_data, strategy_params)
        
        if results is not None:
            all_results.append(results)
    except Exception as e:
        print(f"\nError processing {strategy['file'].name}: {e}")
        continue

print("\n\nProcessing complete!")

if all_results:
    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    
    print(f"\nGenerated {len(results_df)} result rows")
    
    # Overall best configurations
    print("\n" + "=" * 80)
    print("🎯 OPTIMAL STOP/TARGET CONFIGURATIONS")
    print("=" * 80)
    
    # Best overall
    best_overall = results_df[results_df['regime'] == 'Overall'].nlargest(10, 'sharpe_ratio')
    print("\nTop 10 configurations (Overall):")
    for idx, row in best_overall.iterrows():
        print(f"{row['strategy']} | Stop: {row['stop_pct']}%, Target: {row['target_pct']}% → "
              f"Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']*100:.2f}%, "
              f"Win Rate: {row['win_rate']*100:.1f}%")
    
    # Best by regime
    for regime in ['Low Vol', 'Medium Vol', 'High Vol', 'Trending Up', 'Trending Down', 'Ranging']:
        regime_data = results_df[results_df['regime'] == regime]
        if len(regime_data) > 0:
            best_regime = regime_data.nlargest(3, 'sharpe_ratio')
            if len(best_regime) > 0:
                print(f"\nTop configurations for {regime}:")
                for idx, row in best_regime.iterrows():
                    print(f"  Stop: {row['stop_pct']}%, Target: {row['target_pct']}% → "
                          f"Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']*100:.2f}%")
    
    # Check if intraday constraint is respected
    print("\n" + "=" * 80)
    print("🕐 INTRADAY CONSTRAINT CHECK")
    print("=" * 80)
    
    # Sample check on top performers
    constraint_violations = 0
    for idx, strategy in enumerate(strategies_to_analyze[:10]):
        df = pd.read_parquet(strategy['file'])
        df['ts'] = pd.to_datetime(df['ts'])
        df['hour'] = df['ts'].dt.hour
        
        # Check for signals outside market hours (9:30 AM - 4:00 PM ET)
        outside_hours = df[(df['hour'] < 9) | (df['hour'] >= 16) | 
                          ((df['hour'] == 9) & (df['ts'].dt.minute < 30))]
        
        if len(outside_hours[outside_hours['val'] != 0]) > 0:
            constraint_violations += 1
    
    print(f"✅ All top 10 strategies respect intraday constraints!" if constraint_violations == 0 
          else f"⚠️ {constraint_violations}/10 strategies have signals outside market hours")
    
    # Aggregate analysis
    print("\n" + "=" * 80)
    print("📊 AGGREGATE ANALYSIS BY STOP/TARGET")
    print("=" * 80)
    
    # Average performance by stop/target
    agg_overall = results_df[results_df['regime'] == 'Overall'].groupby(['stop_pct', 'target_pct']).agg({
        'sharpe_ratio': ['mean', 'std', 'count'],
        'total_return': ['mean', 'std'],
        'win_rate': 'mean',
        'stop_hits': 'mean',
        'target_hits': 'mean'
    }).round(3)
    
    print("\nAverage performance by stop/target configuration:")
    for (stop, target), row in agg_overall.iterrows():
        print(f"\nStop: {stop}%, Target: {target}%")
        print(f"  Avg Sharpe: {row[('sharpe_ratio', 'mean')]:.2f} (±{row[('sharpe_ratio', 'std')]:.2f})")
        print(f"  Avg Return: {row[('total_return', 'mean')]*100:.2f}% (±{row[('total_return', 'std')]*100:.2f}%)")
        print(f"  Win Rate: {row[('win_rate', 'mean')]*100:.1f}%")
        print(f"  Stop hits: {row[('stop_hits', 'mean')]:.1f}%, Target hits: {row[('target_hits', 'mean')]:.1f}%")
        print(f"  Strategies tested: {row[('sharpe_ratio', 'count')]}")
    
    # Special focus on 0.075/0.1 configuration
    focus_config = results_df[(results_df['stop_pct'] == 0.075) & 
                             (results_df['target_pct'] == 0.1) & 
                             (results_df['regime'] == 'Overall')]
    
    if len(focus_config) > 0:
        print("\n" + "=" * 80)
        print("🎯 FOCUS: 0.075% Stop / 0.1% Target Performance")
        print("=" * 80)
        print(f"Strategies with this config: {len(focus_config)}")
        print(f"Average Sharpe: {focus_config['sharpe_ratio'].mean():.2f}")
        print(f"Average Return: {focus_config['total_return'].mean()*100:.2f}%")
        print(f"Average Win Rate: {focus_config['win_rate'].mean()*100:.1f}%")
        print(f"Average trades: {focus_config['num_trades'].mean():.0f}")
        
        # Top performers with this config
        top_focus = focus_config.nlargest(5, 'sharpe_ratio')
        print("\nTop 5 strategies with 0.075/0.1 config:")
        for idx, row in top_focus.iterrows():
            print(f"  {row['strategy']} → Sharpe: {row['sharpe_ratio']:.2f}, "
                  f"Return: {row['total_return']*100:.2f}%, Trades: {row['num_trades']}")
    
    # Create comprehensive visualizations
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Sharpe heatmap for all regimes
    regime_list = ['Overall', 'Low Vol', 'Medium Vol', 'High Vol', 
                   'Trending Up', 'Trending Down', 'Ranging']
    
    for idx, regime in enumerate(regime_list):
        ax = plt.subplot(3, 3, idx + 1)
        regime_data = results_df[results_df['regime'] == regime]
        
        if len(regime_data) > 0:
            pivot = regime_data.pivot_table(
                values='sharpe_ratio',
                index='stop_pct',
                columns='target_pct',
                aggfunc='mean'
            )
            
            if not pivot.empty:
                sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax,
                           cbar_kws={'label': 'Sharpe Ratio'})
                ax.set_title(f'{regime} - Sharpe by Stop/Target')
                ax.set_xlabel('Target %')
                ax.set_ylabel('Stop %')
                
                # Highlight best combination
                best_idx = regime_data['sharpe_ratio'].idxmax()
                if not pd.isna(best_idx):
                    best_config = regime_data.loc[best_idx]
                    ax.text(0.02, 0.98, f"Best: {best_config['stop_pct']:.3f}/{best_config['target_pct']:.3f}%\nSharpe: {best_config['sharpe_ratio']:.2f}",
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                           fontsize=8)
        else:
            ax.text(0.5, 0.5, f'No data for {regime}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{regime} - No Data')
    
    # 8. Parameter distribution
    ax8 = plt.subplot(3, 3, 8)
    # Show distribution of best stop levels by regime
    best_stops_by_regime = []
    for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
        regime_best = results_df[results_df['regime'] == regime].nlargest(5, 'sharpe_ratio')
        if len(regime_best) > 0:
            best_stops_by_regime.extend([(regime, stop) for stop in regime_best['stop_pct']])
    
    if best_stops_by_regime:
        stop_df = pd.DataFrame(best_stops_by_regime, columns=['Regime', 'Stop %'])
        stop_df.boxplot(column='Stop %', by='Regime', ax=ax8)
        ax8.set_title('Optimal Stop % Distribution by Volatility')
    
    # 9. Summary text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"Analysis Summary\n\n"
    summary_text += f"Strategies analyzed: {len(strategies_to_analyze)}\n"
    summary_text += f"Stop/Target configs: {len(STOP_TARGET_CONFIGS)}\n"
    summary_text += f"Total results: {len(results_df)}\n"
    summary_text += f"Trading period: {actual_trading_days} days\n\n"
    
    # Best overall
    if len(best_overall) > 0:
        best = best_overall.iloc[0]
        summary_text += f"Best Overall Config:\n"
        summary_text += f"  Strategy: {best['strategy']}\n"
        summary_text += f"  Stop/Target: {best['stop_pct']}%/{best['target_pct']}%\n"
        summary_text += f"  Sharpe: {best['sharpe_ratio']:.2f}\n"
        summary_text += f"  Return: {best['total_return']*100:.2f}%\n"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            verticalalignment='top', fontsize=12, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    output_path = RESULTS_DIR / 'comprehensive_bollinger_analysis.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")
    
    # Save best configurations
    best_configs = results_df[results_df['regime'] == 'Overall'].nlargest(20, 'sharpe_ratio')
    best_configs.to_csv(RESULTS_DIR / 'best_bollinger_configs.csv', index=False)
    
else:
    print("\n❌ No valid results generated!")

print("\n" + "=" * 80)
print("💡 Next Steps:")
print("1. Review regime-specific configurations")
print("2. Implement adaptive stop/target based on current regime")
print("3. Consider tighter stops for 1-minute vs 5-minute data")
print("4. Test on out-of-sample data with regime adaptation")
print("=" * 80)