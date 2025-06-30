# Equity Curve Analysis - Build and visualize equity curves overlaid with SPY
# Shows cumulative returns, drawdowns, and performance metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path('/Users/daws/ADMF-PC/config/bollinger/results/20250625_173629')
SIGNAL_DIR = RESULTS_DIR / 'traces/signals/bollinger_bands'
DATA_DIR = Path('/Users/daws/ADMF-PC/data')

# Stop/Target configurations to compare
CONFIGS_TO_COMPARE = [
    (0, 0, "No Stop/Target"),           # Baseline
    (0.075, 0.10, "0.075% Stop / 0.1% Target"),  # Proven optimal
    (0.05, 0.075, "0.05% Stop / 0.075% Target"), # Tighter
    (0.10, 0.15, "0.1% Stop / 0.15% Target"),    # Wider
]

print("📈 Equity Curve Analysis with SPY Overlay")
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

# Calculate SPY returns for comparison
market_data['spy_return'] = market_data[close_col].pct_change()
market_data['spy_cumulative'] = (1 + market_data['spy_return']).cumprod()

def build_equity_curve(signal_file, market_data, stop_pct, target_pct, execution_cost_bps=1.0):
    """Build equity curve from signals with stop/target logic"""
    
    # Load signals
    df = pd.read_parquet(signal_file)
    df['ts'] = pd.to_datetime(df['ts'])
    if hasattr(df['ts'].dtype, 'tz'):
        df['ts'] = df['ts'].dt.tz_localize(None)
    
    df = df.sort_values('ts')
    
    # Initialize equity curve
    equity_curve = []
    current_equity = 100000  # Start with $100k
    current_pos = 0
    entry = None
    trades = []
    
    for idx, row in df.iterrows():
        signal = row['val']
        
        if current_pos == 0 and signal != 0:
            # Entry
            current_pos = signal
            entry = {
                'time': row['ts'], 
                'price': row['px'], 
                'direction': signal,
                'equity_at_entry': current_equity
            }
            
        elif current_pos != 0 and signal != current_pos:
            # Exit - check for stop/target
            if entry:
                # Find market data between entry and current signal
                mask = (market_data['timestamp'] >= entry['time']) & (market_data['timestamp'] <= row['ts'])
                trade_bars = market_data[mask]
                
                exit_price = row['px']
                exit_time = row['ts']
                exit_type = 'signal'
                
                if len(trade_bars) > 0 and (stop_pct > 0 or target_pct > 0):
                    entry_price = entry['price']
                    
                    if entry['direction'] > 0:  # Long
                        stop_price = entry_price * (1 - stop_pct/100) if stop_pct > 0 else 0
                        target_price = entry_price * (1 + target_pct/100) if target_pct > 0 else float('inf')
                        
                        for bar_idx, bar in trade_bars.iterrows():
                            if stop_pct > 0 and bar[low_col] <= stop_price:
                                exit_price = stop_price
                                exit_time = bar['timestamp']
                                exit_type = 'stop'
                                break
                            elif target_pct > 0 and bar[high_col] >= target_price:
                                exit_price = target_price
                                exit_time = bar['timestamp']
                                exit_type = 'target'
                                break
                                
                    else:  # Short
                        stop_price = entry_price * (1 + stop_pct/100) if stop_pct > 0 else float('inf')
                        target_price = entry_price * (1 - target_pct/100) if target_pct > 0 else 0
                        
                        for bar_idx, bar in trade_bars.iterrows():
                            if stop_pct > 0 and bar[high_col] >= stop_price:
                                exit_price = stop_price
                                exit_time = bar['timestamp']
                                exit_type = 'stop'
                                break
                            elif target_pct > 0 and bar[low_col] <= target_price:
                                exit_price = target_price
                                exit_time = bar['timestamp']
                                exit_type = 'target'
                                break
                
                # Calculate P&L
                if entry['direction'] > 0:
                    raw_return = (exit_price - entry['price']) / entry['price']
                else:
                    raw_return = (entry['price'] - exit_price) / entry['price']
                
                # Apply execution cost
                net_return = raw_return - (execution_cost_bps * 2 / 10000)
                
                # Update equity
                position_size = entry['equity_at_entry']  # Use full equity (can adjust for position sizing)
                pnl = position_size * net_return
                current_equity += pnl
                
                # Record trade
                trades.append({
                    'entry_time': entry['time'],
                    'exit_time': exit_time,
                    'entry_price': entry['price'],
                    'exit_price': exit_price,
                    'direction': entry['direction'],
                    'net_return': net_return,
                    'pnl': pnl,
                    'equity_after': current_equity,
                    'exit_type': exit_type
                })
                
                # Record equity point
                equity_curve.append({
                    'timestamp': exit_time,
                    'equity': current_equity,
                    'trade_count': len(trades),
                    'position': 0  # Flat after exit
                })
            
            # Update position
            current_pos = signal
            if signal != 0:
                entry = {
                    'time': row['ts'], 
                    'price': row['px'], 
                    'direction': signal,
                    'equity_at_entry': current_equity
                }
                # Record position entry
                equity_curve.append({
                    'timestamp': row['ts'],
                    'equity': current_equity,
                    'trade_count': len(trades),
                    'position': signal
                })
            else:
                entry = None
    
    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

# Load strategy index to find a good strategy
strategy_index = pd.read_parquet(RESULTS_DIR / 'strategy_index.parquet')

# Find a strategy with good signal count
print("\nFinding strategy with good signal count...")
best_strategy = None
signal_counts = []

for signal_file in SIGNAL_DIR.glob('*.parquet'):
    df = pd.read_parquet(signal_file)
    non_zero = (df['val'] != 0).sum()
    signal_counts.append(non_zero)
    
    # Lower threshold for test set
    if non_zero > 200:  # Adjusted for test set size
        strategy_num = int(signal_file.stem.split('_')[-1])
        if strategy_num < len(strategy_index):
            params = strategy_index.iloc[strategy_num]
            best_strategy = {
                'file': signal_file,
                'period': params.get('period', 'N/A'),
                'std_dev': params.get('std_dev', 'N/A'),
                'signal_count': non_zero
            }
            if non_zero > 500:  # If we find a really good one, use it
                break

# Debug info
print(f"Signal count distribution: min={min(signal_counts)}, max={max(signal_counts)}, mean={np.mean(signal_counts):.0f}")
print(f"Strategies with >100 signals: {sum(1 for s in signal_counts if s > 100)}")
print(f"Strategies with >200 signals: {sum(1 for s in signal_counts if s > 200)}")
print(f"Strategies with >500 signals: {sum(1 for s in signal_counts if s > 500)}")

if best_strategy is None:
    print("❌ No strategy with sufficient signals found!")
    import sys
    sys.exit()

print(f"\nUsing strategy: Period={best_strategy['period']}, StdDev={best_strategy['std_dev']} "
      f"({best_strategy['signal_count']} signals)")

# Build equity curves for different configurations
fig, axes = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 1, 1])

# Store results for comparison
all_results = {}

for stop_pct, target_pct, label in CONFIGS_TO_COMPARE:
    print(f"\nBuilding equity curve for {label}...")
    
    equity_df, trades_df = build_equity_curve(
        best_strategy['file'],
        market_data,
        stop_pct,
        target_pct,
        execution_cost_bps=1.0
    )
    
    if len(equity_df) == 0:
        print(f"  No trades generated!")
        continue
    
    # Merge with market data to align timestamps
    equity_df = equity_df.set_index('timestamp')
    
    # Calculate performance metrics
    initial_equity = 100000
    final_equity = equity_df['equity'].iloc[-1]
    total_return = (final_equity - initial_equity) / initial_equity
    
    # Calculate drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = equity_df['drawdown'].min()
    
    # Win rate
    if len(trades_df) > 0:
        win_rate = (trades_df['net_return'] > 0).mean()
        avg_win = trades_df[trades_df['net_return'] > 0]['net_return'].mean() if (trades_df['net_return'] > 0).any() else 0
        avg_loss = trades_df[trades_df['net_return'] <= 0]['net_return'].mean() if (trades_df['net_return'] <= 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Exit type breakdown
        exit_counts = trades_df['exit_type'].value_counts()
    else:
        win_rate = 0
        profit_factor = 0
        exit_counts = pd.Series()
    
    # Store results
    all_results[label] = {
        'equity_df': equity_df,
        'trades_df': trades_df,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'exit_counts': exit_counts,
        'num_trades': len(trades_df)
    }
    
    print(f"  Return: {total_return*100:.2f}%, Max DD: {max_drawdown*100:.2f}%, "
          f"Trades: {len(trades_df)}, Win Rate: {win_rate*100:.1f}%")

# Plot 1: Equity curves with SPY overlay
ax1 = axes[0]

# Plot SPY (normalized to start at 100k)
spy_start_value = 100000
spy_normalized = market_data[close_col] / market_data[close_col].iloc[0] * spy_start_value
ax1.plot(market_data['timestamp'], spy_normalized, 'gray', alpha=0.5, linewidth=2, label='SPY')

# Plot equity curves
colors = ['red', 'green', 'blue', 'orange', 'purple']
for i, (label, results) in enumerate(all_results.items()):
    equity_df = results['equity_df']
    ax1.plot(equity_df.index, equity_df['equity'], colors[i], 
             linewidth=2, label=f"{label} ({results['total_return']*100:.1f}%)")

ax1.set_ylabel('Portfolio Value ($)')
ax1.set_title(f'Equity Curves: Bollinger Strategy (P={best_strategy["period"]}, S={best_strategy["std_dev"]}) vs Buy & Hold SPY')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Add vertical lines for major market events (optional)
# You could add lines for specific dates if needed

# Plot 2: Drawdown comparison
ax2 = axes[1]

for i, (label, results) in enumerate(all_results.items()):
    if len(results['equity_df']) > 0:
        ax2.fill_between(results['equity_df'].index, 
                        results['equity_df']['drawdown'] * 100, 
                        0, alpha=0.5, color=colors[i], label=label)

ax2.set_ylabel('Drawdown %')
ax2.set_title('Drawdown Comparison')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(top=0)

# Plot 3: Position overlay - show when strategy is long/short/flat
ax3 = axes[2]

# Use the best performing configuration for position display
best_config_label = max(all_results.keys(), 
                       key=lambda k: all_results[k]['total_return'] if all_results[k]['num_trades'] > 0 else -999)
best_equity_df = all_results[best_config_label]['equity_df']

if 'position' in best_equity_df.columns:
    # Create position visualization
    position_data = best_equity_df['position'].fillna(0)
    
    # Color based on position
    for i in range(len(position_data)-1):
        if position_data.iloc[i] > 0:  # Long
            ax3.axvspan(position_data.index[i], position_data.index[i+1], 
                       alpha=0.3, color='green')
        elif position_data.iloc[i] < 0:  # Short
            ax3.axvspan(position_data.index[i], position_data.index[i+1], 
                       alpha=0.3, color='red')
    
    # Also plot SPY price for reference
    ax3_twin = ax3.twinx()
    ax3_twin.plot(market_data['timestamp'], market_data[close_col], 
                  'black', alpha=0.7, linewidth=1)
    ax3_twin.set_ylabel('SPY Price ($)')
    
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Short', 'Flat', 'Long'])
    ax3.set_ylabel('Position')
    ax3.set_title(f'Trading Position Over Time ({best_config_label})')
    ax3.grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# Summary statistics table
print("\n" + "=" * 80)
print("📊 PERFORMANCE SUMMARY")
print("=" * 80)

# Create comparison table
summary_data = []
for label, results in all_results.items():
    if results['num_trades'] > 0:
        summary_data.append({
            'Configuration': label,
            'Total Return': f"{results['total_return']*100:.2f}%",
            'Max Drawdown': f"{results['max_drawdown']*100:.2f}%",
            'Sharpe Ratio': 'N/A',  # Would need to calculate
            'Win Rate': f"{results['win_rate']*100:.1f}%",
            'Profit Factor': f"{results['profit_factor']:.2f}",
            'Total Trades': results['num_trades'],
            'Avg Trades/Day': f"{results['num_trades'] / ((equity_df.index[-1] - equity_df.index[0]).days / 365 * 252):.1f}"
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Exit type breakdown for best configuration
print(f"\n📈 EXIT TYPE BREAKDOWN - {best_config_label}")
print("=" * 60)

best_exit_counts = all_results[best_config_label]['exit_counts']
if len(best_exit_counts) > 0:
    total_exits = best_exit_counts.sum()
    for exit_type, count in best_exit_counts.items():
        print(f"{exit_type.capitalize():10s}: {count:4d} ({count/total_exits*100:5.1f}%)")

# Monthly returns analysis
print("\n📅 MONTHLY RETURNS - Best Configuration")
print("=" * 60)

best_trades = all_results[best_config_label]['trades_df']
if len(best_trades) > 0:
    best_trades['month'] = pd.to_datetime(best_trades['exit_time']).dt.to_period('M')
    monthly_returns = best_trades.groupby('month').agg({
        'net_return': ['sum', 'count', 'mean'],
        'pnl': 'sum'
    })
    
    print("Month     | Return  | Trades | Avg/Trade |    P&L")
    print("-" * 60)
    for month, row in monthly_returns.iterrows():
        print(f"{month} | {row[('net_return', 'sum')]*100:6.2f}% | "
              f"{row[('net_return', 'count')]:6.0f} | {row[('net_return', 'mean')]*100:7.3f}% | "
              f"${row[('pnl', 'sum')]:8,.0f}")

# Risk metrics
print("\n⚠️ RISK METRICS")
print("=" * 60)

for label, results in all_results.items():
    if results['num_trades'] > 0:
        equity_curve = results['equity_df']['equity']
        returns = equity_curve.pct_change().dropna()
        
        # Calculate metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Calmar ratio (return / max drawdown)
        calmar = results['total_return'] / abs(results['max_drawdown']) if results['max_drawdown'] != 0 else 0
        
        print(f"\n{label}:")
        print(f"  Volatility (ann.): {volatility*100:.1f}%")
        print(f"  Downside Vol (ann.): {downside_vol*100:.1f}%")
        print(f"  Calmar Ratio: {calmar:.2f}")
        print(f"  Max Consecutive Losses: ", end='')
        
        # Count max consecutive losses
        if len(results['trades_df']) > 0:
            losses = (results['trades_df']['net_return'] < 0).astype(int)
            max_consec_losses = losses.groupby((losses != losses.shift()).cumsum()).sum().max()
            print(f"{max_consec_losses}")
        else:
            print("N/A")

print("\n" + "=" * 80)
print("💡 KEY INSIGHTS")
print("=" * 80)

# Compare to SPY
spy_return = (market_data[close_col].iloc[-1] - market_data[close_col].iloc[0]) / market_data[close_col].iloc[0]
print(f"\nSPY Buy & Hold Return: {spy_return*100:.2f}%")

best_return = max(all_results.values(), key=lambda x: x['total_return'])['total_return']
print(f"Best Strategy Return: {best_return*100:.2f}%")
print(f"Outperformance: {(best_return - spy_return)*100:.2f}%")

# Save detailed results
output_path = RESULTS_DIR / 'equity_curve_analysis.csv'
all_trades = []
for label, results in all_results.items():
    if len(results['trades_df']) > 0:
        trades = results['trades_df'].copy()
        trades['configuration'] = label
        all_trades.append(trades)

if all_trades:
    combined_trades = pd.concat(all_trades, ignore_index=True)
    combined_trades.to_csv(output_path, index=False)
    print(f"\n✅ Detailed trades saved to: {output_path}")

print("\n" + "=" * 80)