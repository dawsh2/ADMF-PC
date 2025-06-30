# Comprehensive 1-Minute Bollinger Analysis
# Identifies top strategies, optimal stops/targets, and regime-specific configurations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check if required variables exist, define if missing
if 'execution_cost_bps' not in globals():
    execution_cost_bps = 1.0  # 1 basis point default
    print("Set execution_cost_bps = 1.0")

if 'run_dir' not in globals():
    print("⚠️ run_dir not found - using current directory")
    run_dir = Path.cwd()

# Quick check for required variables
print("Checking prerequisites...")
print(f"✓ market_data exists: {'market_data' in globals()}")
print(f"✓ performance_df exists: {'performance_df' in globals()}")
print(f"✓ execution_cost_bps: {execution_cost_bps} bps")
print(f"✓ run_dir: {run_dir}")
print(f"✓ extract_trades exists: {'extract_trades' in globals()}")
print(f"✓ apply_stop_target exists: {'apply_stop_target' in globals()}")

# Define extract_trades if not available
if 'extract_trades' not in globals():
    def extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
        """Extract individual trades from strategy signals."""
        trace_df = pd.read_parquet(trace_path)
        strategy_signals = trace_df[trace_df['strategy_hash'] == strategy_hash].copy()
        
        if len(strategy_signals) == 0:
            return pd.DataFrame()
        
        strategy_signals = strategy_signals.sort_values('ts')
        
        # Work with a copy to avoid modifying the original
        market_data_copy = market_data.copy()
        
        # Ensure timestamps are properly formatted for comparison
        # Convert market_data timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(market_data_copy['timestamp']):
            market_data_copy['timestamp'] = pd.to_datetime(market_data_copy['timestamp'])
        
        # Convert signal timestamps to datetime if needed
        # Note: timestamps in parquet files are stored as strings
        strategy_signals['ts'] = pd.to_datetime(strategy_signals['ts'])
        
        # Remove timezones from both for consistent comparison
        # Check if timezone-aware and remove timezone
        if hasattr(market_data_copy['timestamp'].dtype, 'tz') and market_data_copy['timestamp'].dt.tz is not None:
            market_data_copy['timestamp'] = market_data_copy['timestamp'].dt.tz_localize(None)
        
        if hasattr(strategy_signals['ts'].dtype, 'tz') and strategy_signals['ts'].dt.tz is not None:
            strategy_signals['ts'] = strategy_signals['ts'].dt.tz_localize(None)
        
        trades = []
        current_position = 0
        entry_idx = None
        entry_price = None
        entry_time = None
        
        for idx, signal in strategy_signals.iterrows():
            signal_value = signal['val']
            timestamp = signal['ts']
            
            # Find closest timestamp match (within 1 minute tolerance)
            time_diff = (market_data_copy['timestamp'] - timestamp).abs()
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx] <= pd.Timedelta(minutes=1):
                market_idx = [closest_idx]
            else:
                market_idx = []
            if len(market_idx) == 0:
                continue
            market_idx = market_idx[0]
            
            # Handle both 'close' and 'Close' column names
            close_col = 'Close' if 'Close' in market_data_copy.columns else 'close'
            current_price = market_data_copy.loc[market_idx, close_col]
            
            if current_position == 0 and signal_value != 0:
                current_position = signal_value
                entry_idx = market_idx
                entry_price = current_price
                entry_time = timestamp
                
            elif current_position != 0 and signal_value != current_position:
                if entry_idx is not None:
                    exit_idx = market_idx
                    exit_price = current_price
                    exit_time = timestamp
                    
                    direction = 1 if current_position > 0 else -1
                    raw_return = direction * (exit_price - entry_price) / entry_price
                    execution_cost = execution_cost_bps / 10000
                    net_return = raw_return - execution_cost
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_idx': entry_idx,
                        'exit_idx': exit_idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': direction,
                        'raw_return': raw_return,
                        'execution_cost': execution_cost,
                        'net_return': net_return
                    })
                
                if signal_value != 0:
                    current_position = signal_value
                    entry_idx = market_idx
                    entry_price = current_price
                    entry_time = timestamp
                else:
                    current_position = 0
                    entry_idx = None
                    entry_price = None
                    entry_time = None
        
        return pd.DataFrame(trades)
    print("✓ Defined extract_trades function")

# Define apply_stop_target if not available
if 'apply_stop_target' not in globals():
    def apply_stop_target(trades_df, stop_pct, target_pct, market_data):
        """Apply stop loss and profit target to trades"""
        if stop_pct == 0 and target_pct == 0:
            return trades_df['net_return'].values, {'stop': 0, 'target': 0, 'signal': len(trades_df)}
        
        modified_returns = []
        exit_types = {'stop': 0, 'target': 0, 'signal': 0}
        
        for _, trade in trades_df.iterrows():
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
            
            if len(trade_prices) == 0:
                modified_returns.append(trade['net_return'])
                exit_types['signal'] += 1
                continue
            
            entry_price = trade['entry_price']
            direction = trade['direction']
            
            if direction == 1:  # Long
                stop_price = entry_price * (1 - stop_pct/100) if stop_pct > 0 else 0
                target_price = entry_price * (1 + target_pct/100)
            else:  # Short
                stop_price = entry_price * (1 + stop_pct/100) if stop_pct > 0 else float('inf')
                target_price = entry_price * (1 - target_pct/100)
            
            exit_price = trade['exit_price']
            exit_type = 'signal'
            
            for _, bar in trade_prices.iterrows():
                if direction == 1:  # Long
                    if stop_pct > 0 and bar['low'] <= stop_price:
                        exit_price = stop_price
                        exit_type = 'stop'
                        break
                    elif target_pct > 0 and bar['high'] >= target_price:
                        exit_price = target_price
                        exit_type = 'target'
                        break
                else:  # Short
                    if stop_pct > 0 and bar['high'] >= stop_price:
                        exit_price = stop_price
                        exit_type = 'stop'
                        break
                    elif target_pct > 0 and bar['low'] <= target_price:
                        exit_price = target_price
                        exit_type = 'target'
                        break
            
            exit_types[exit_type] += 1
            
            if direction == 1:
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price
            
            net_return = raw_return - trade['execution_cost']
            modified_returns.append(net_return)
        
        return np.array(modified_returns), exit_types
    print("✓ Defined apply_stop_target function")

# Check if we have the minimum required data
if 'market_data' not in globals() or 'performance_df' not in globals():
    print("\n❌ ERROR: Missing required data!")
    print("Please ensure you have loaded:")
    print("1. market_data - your 1-minute SPY data")
    print("2. performance_df - results from your backtest")
    raise ValueError("Missing required data")

print("\n✅ All prerequisites satisfied! Ready to run analysis.")

# Clean market_data timestamps if needed
if 'market_data' in globals() and 'timestamp' in market_data.columns:
    # Check if already datetime
    if pd.api.types.is_datetime64_any_dtype(market_data['timestamp']):
        # If timezone-aware, remove timezone
        if market_data['timestamp'].dt.tz is not None:
            print("Removing timezone from market_data timestamps...")
            market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(None)
    else:
        # Convert to datetime, handling timezone-aware strings
        print("Converting market_data timestamps to datetime...")
        try:
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], utc=True).dt.tz_localize(None)
        except:
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])

# Configuration
ANALYZE_ALL_STRATEGIES = True  # Set to True to analyze ALL strategies (can be slow)
TOP_N_STRATEGIES = 20  # Number of top strategies to analyze if ANALYZE_ALL_STRATEGIES is False

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

def calculate_market_regimes(market_data):
    """Calculate volatility and trend regimes"""
    # Determine the close column name (handle both 'close' and 'Close')
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

def analyze_strategy_with_regimes(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """Analyze strategy performance across different regimes and stop/target configs"""
    
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    # Add regime information to trades
    trades_with_regime = trades.merge(
        market_data[['vol_regime', 'trend_regime']], 
        left_on='entry_idx', 
        right_index=True, 
        how='left'
    )
    
    results = []
    
    # Test each stop/target configuration
    for stop_pct, target_pct in STOP_TARGET_CONFIGS:
        # Overall performance
        returns_array, exit_types = apply_stop_target(trades, stop_pct, target_pct, market_data)
        
        if len(returns_array) > 0:
            overall_stats = calculate_performance_metrics(returns_array, len(trades))
            overall_stats['stop_pct'] = stop_pct
            overall_stats['target_pct'] = target_pct
            overall_stats['regime'] = 'Overall'
            results.append(overall_stats)
            
            # Performance by volatility regime
            for vol_regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                regime_trades = trades_with_regime[trades_with_regime['vol_regime'] == vol_regime]
                if len(regime_trades) >= 10:  # Minimum trades for statistics
                    regime_returns, regime_exits = apply_stop_target(regime_trades, stop_pct, target_pct, market_data)
                    regime_stats = calculate_performance_metrics(regime_returns, len(regime_trades))
                    regime_stats['stop_pct'] = stop_pct
                    regime_stats['target_pct'] = target_pct
                    regime_stats['regime'] = vol_regime
                    results.append(regime_stats)
            
            # Performance by trend regime
            for trend_regime in ['Trending Up', 'Trending Down', 'Ranging']:
                regime_trades = trades_with_regime[trades_with_regime['trend_regime'] == trend_regime]
                if len(regime_trades) >= 10:
                    regime_returns, regime_exits = apply_stop_target(regime_trades, stop_pct, target_pct, market_data)
                    regime_stats = calculate_performance_metrics(regime_returns, len(regime_trades))
                    regime_stats['stop_pct'] = stop_pct
                    regime_stats['target_pct'] = target_pct
                    regime_stats['regime'] = trend_regime
                    results.append(regime_stats)
    
    return pd.DataFrame(results)

def calculate_performance_metrics(returns_array, num_trades):
    """Calculate comprehensive performance metrics"""
    
    if len(returns_array) == 0:
        return {}
    
    metrics = {
        'total_return': (1 + returns_array).prod() - 1,
        'avg_return': returns_array.mean(),
        'win_rate': (returns_array > 0).mean(),
        'num_trades': num_trades,
        'sharpe_ratio': 0,
        'profit_factor': 0,
        'max_drawdown': calculate_max_drawdown(returns_array)
    }
    
    # Sharpe ratio (annualized for 1-minute data)
    if returns_array.std() > 0:
        # Assuming ~390 1-minute bars per day
        metrics['sharpe_ratio'] = returns_array.mean() / returns_array.std() * np.sqrt(252 * 390)
    
    # Profit factor
    winners = returns_array[returns_array > 0]
    losers = returns_array[returns_array < 0]
    if len(losers) > 0 and losers.sum() != 0:
        metrics['profit_factor'] = winners.sum() / abs(losers.sum())
    elif len(winners) > 0:
        metrics['profit_factor'] = np.inf
    
    return metrics

def calculate_max_drawdown(returns_array):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns_array).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# Main Analysis
print("📊 Comprehensive 1-Minute Bollinger Analysis")
print("=" * 80)

# Ensure market data timestamp is datetime
if 'timestamp' in market_data.columns:
    # Check if it's already datetime
    if not pd.api.types.is_datetime64_any_dtype(market_data['timestamp']):
        print("Converting market data timestamps to datetime...")
        try:
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'], utc=True)
        except:
            # If that fails, try without UTC
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
    
    # If timezone-aware, remove timezone for consistency
    if market_data['timestamp'].dt.tz is not None:
        print("Removing timezone from market data timestamps for analysis...")
        market_data['timestamp'] = market_data['timestamp'].dt.tz_localize(None)

# First, add regime calculations to market data
print("Calculating market regimes...")
market_data = calculate_market_regimes(market_data)

# Get actual trading days
if 'performance_df' in globals() and len(performance_df) > 0:
    # Check for trace data to get actual date range
    if 'trace_path' in performance_df.columns:
        try:
            sample_trace = pd.read_parquet(performance_df.iloc[0]['trace_path'])
            if 'ts' in sample_trace.columns:
                actual_start = pd.to_datetime(sample_trace['ts'].min())
                actual_end = pd.to_datetime(sample_trace['ts'].max())
                actual_trading_days = len(pd.bdate_range(actual_start, actual_end))
                print(f"Actual trading period: {actual_start.date()} to {actual_end.date()} ({actual_trading_days} days)")
            else:
                actual_trading_days = len(market_data['timestamp'].dt.date.unique())
        except:
            actual_trading_days = len(market_data['timestamp'].dt.date.unique())
    else:
        actual_trading_days = len(market_data['timestamp'].dt.date.unique())
    
    # Basic statistics
    print(f"\nTotal strategies: {len(performance_df)}")
    performance_df['trades_per_day'] = performance_df['num_trades'] / actual_trading_days
    
    # Filter for meaningful analysis - adjust based on your data
    # For shorter test periods, use a lower threshold
    min_trades = 20  # Lowered from 100 for shorter test periods
    valid_strategies = performance_df[performance_df['num_trades'] >= min_trades].copy()
    print(f"Strategies with >= {min_trades} trades: {len(valid_strategies)}")
    
    # If still no strategies, use an even lower threshold
    if len(valid_strategies) == 0:
        min_trades = 10
        valid_strategies = performance_df[performance_df['num_trades'] >= min_trades].copy()
        print(f"Lowered threshold: Strategies with >= {min_trades} trades: {len(valid_strategies)}")
        
    # If STILL no strategies, just use all strategies
    if len(valid_strategies) == 0:
        print("⚠️ Very few trades per strategy. Using all strategies.")
        valid_strategies = performance_df.copy()
        
    # Show trade statistics
    print(f"\nTrade statistics:")
    print(f"  Mean trades per strategy: {performance_df['num_trades'].mean():.1f}")
    print(f"  Max trades: {performance_df['num_trades'].max()}")
    print(f"  Min trades: {performance_df['num_trades'].min()}")
    print(f"  Trading days: {actual_trading_days}")
    
    # Show distribution of trades
    trade_bins = [0, 5, 10, 20, 50, 100, 200, 500, 1000]
    trade_counts = pd.cut(performance_df['num_trades'], bins=trade_bins).value_counts().sort_index()
    print("\nTrade count distribution:")
    for interval, count in trade_counts.items():
        print(f"  {interval}: {count} strategies")
    
    # Show top 10 by trade count
    print("\nTop 10 strategies by trade count:")
    top_by_trades = performance_df.nlargest(10, 'num_trades')
    for i, (_, row) in enumerate(top_by_trades.iterrows()):
        params = []
        if 'period' in row and pd.notna(row['period']):
            params.append(f"period={row['period']}")
        if 'std_dev' in row and pd.notna(row['std_dev']):
            params.append(f"std_dev={row['std_dev']}")
        param_str = ', '.join(params) if params else 'N/A'
        print(f"  {i+1}. {row['strategy_type']} ({param_str}): {row['num_trades']} trades, {row['trades_per_day']:.2f}/day")
    
    # Check why trades might be low
    if performance_df['num_trades'].max() < 50:
        print("\n⚠️ Low trade counts detected. Possible reasons:")
        print("  - Short test period (only {actual_trading_days} days)")
        print("  - Conservative Bollinger Band parameters")
        print("  - 1-minute bars may need tighter bands (lower std_dev)")
        print("  - Consider testing with more aggressive parameters")
    
    if len(valid_strategies) > 0:
        # Sort by base Sharpe ratio
        valid_strategies = valid_strategies.sort_values('sharpe_ratio', ascending=False)
        
        # Determine how many strategies to analyze
        if ANALYZE_ALL_STRATEGIES:
            strategies_to_analyze = valid_strategies
            print(f"\nAnalyzing ALL {len(valid_strategies)} strategies (this may take a while)...")
        else:
            TOP_N = min(TOP_N_STRATEGIES, len(valid_strategies))
            strategies_to_analyze = valid_strategies.head(TOP_N)
            print(f"\nAnalyzing top {TOP_N} strategies by base Sharpe ratio...")
            print(f"(Set ANALYZE_ALL_STRATEGIES=True to analyze all {len(valid_strategies)} strategies)")
        
        all_results = []
        
        for idx, (_, strategy) in enumerate(strategies_to_analyze.iterrows()):
            print(f"\rProcessing strategy {idx+1}/{len(strategies_to_analyze)}...", end='', flush=True)
            
            # Analyze with regime-specific configurations
            # Ensure full path to trace file
            full_trace_path = run_dir / strategy['trace_path']
            regime_results = analyze_strategy_with_regimes(
                strategy['strategy_hash'],
                str(full_trace_path),
                market_data,
                execution_cost_bps
            )
            
            if regime_results is not None:
                regime_results['strategy_hash'] = strategy['strategy_hash']
                regime_results['period'] = strategy.get('period', 'N/A')
                regime_results['std_dev'] = strategy.get('std_dev', 'N/A')
                regime_results['base_sharpe'] = strategy['sharpe_ratio']
                regime_results['base_return'] = strategy['total_return']
                all_results.append(regime_results)
        
        print("\n\nProcessing complete!")
        
        if all_results:
            # Combine all results
            results_df = pd.concat(all_results, ignore_index=True)
            
            # Find optimal configurations
            print("\n" + "="*80)
            print("🏆 OPTIMAL CONFIGURATIONS BY REGIME")
            print("="*80)
            
            # Overall best
            overall_best = results_df[results_df['regime'] == 'Overall'].nlargest(10, 'sharpe_ratio')
            print("\n1. Overall Best Configurations:")
            print("-" * 60)
            for _, row in overall_best.head(5).iterrows():
                print(f"\nStrategy: period={row['period']}, std_dev={row['std_dev']}")
                print(f"Stop/Target: {row['stop_pct']}%/{row['target_pct']}%")
                print(f"Sharpe: {row['sharpe_ratio']:.2f} (base: {row['base_sharpe']:.2f})")
                print(f"Return: {row['total_return']*100:.2f}% (base: {row['base_return']*100:.2f}%)")
                print(f"Win Rate: {row['win_rate']*100:.1f}%")
            
            # Best by volatility regime
            print("\n2. Optimal by Volatility Regime:")
            print("-" * 60)
            for vol_regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                regime_best = results_df[results_df['regime'] == vol_regime].nlargest(3, 'sharpe_ratio')
                if len(regime_best) > 0:
                    print(f"\n{vol_regime}:")
                    best = regime_best.iloc[0]
                    print(f"  Best: {best['stop_pct']}%/{best['target_pct']}% stop/target")
                    print(f"  Sharpe: {best['sharpe_ratio']:.2f}, Return: {best['total_return']*100:.2f}%")
                    print(f"  Trades: {best['num_trades']:.0f}")
            
            # Best by trend regime
            print("\n3. Optimal by Trend Regime:")
            print("-" * 60)
            for trend_regime in ['Trending Up', 'Trending Down', 'Ranging']:
                regime_best = results_df[results_df['regime'] == trend_regime].nlargest(3, 'sharpe_ratio')
                if len(regime_best) > 0:
                    print(f"\n{trend_regime}:")
                    best = regime_best.iloc[0]
                    print(f"  Best: {best['stop_pct']}%/{best['target_pct']}% stop/target")
                    print(f"  Sharpe: {best['sharpe_ratio']:.2f}, Return: {best['total_return']*100:.2f}%")
                    print(f"  Trades: {best['num_trades']:.0f}")
            
            # Create separate heatmaps for each regime
            # First, create a figure for regime heatmaps
            fig_regimes = plt.figure(figsize=(20, 16))
            
            # Volatility regime heatmaps
            regime_list = ['Overall', 'Low Vol', 'Medium Vol', 'High Vol', 'Trending Up', 'Trending Down', 'Ranging']
            
            for idx, regime in enumerate(regime_list):
                ax = plt.subplot(3, 3, idx + 1)
                regime_data = results_df[results_df['regime'] == regime]
                
                if len(regime_data) > 0:
                    # Create pivot table for heatmap
                    pivot = regime_data.pivot_table(
                        values='sharpe_ratio',
                        index='stop_pct',
                        columns='target_pct',
                        aggfunc='mean'
                    )
                    
                    # Create heatmap
                    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax,
                               cbar_kws={'label': 'Sharpe Ratio'})
                    ax.set_title(f'{regime} - Sharpe by Stop/Target')
                    ax.set_xlabel('Target %')
                    ax.set_ylabel('Stop %')
                    
                    # Highlight best combination
                    best_idx = regime_data['sharpe_ratio'].idxmax()
                    best_config = regime_data.loc[best_idx]
                    ax.text(0.02, 0.98, f"Best: {best_config['stop_pct']:.3f}/{best_config['target_pct']:.2f}%\nSharpe: {best_config['sharpe_ratio']:.2f}",
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                           fontsize=8)
                else:
                    ax.text(0.5, 0.5, f'No data for {regime}', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{regime} - No Data')
            
            # Add parameter distribution plots
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
            
            # Add summary text
            ax9 = plt.subplot(3, 3, 9)
            ax9.axis('off')
            
            summary_text = "Parameter Heatmap Summary\n\n"
            for regime in regime_list[1:]:  # Skip 'Overall'
                regime_best = results_df[results_df['regime'] == regime].nlargest(1, 'sharpe_ratio')
                if len(regime_best) > 0:
                    best = regime_best.iloc[0]
                    summary_text += f"{regime}: {best['stop_pct']:.3f}/{best['target_pct']:.2f}%\n"
            
            ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                    verticalalignment='top', fontsize=11, family='monospace')
            
            plt.tight_layout()
            plt.savefig(run_dir / f'regime_heatmaps_{timestamp}.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # Now create the original summary figure
            fig = plt.figure(figsize=(20, 12))
            
            # 1. Overall performance comparison
            ax1 = plt.subplot(2, 3, 1)
            # Show top configs across all regimes
            top_overall = results_df.nlargest(15, 'sharpe_ratio')
            regime_colors = {'Overall': 'blue', 'Low Vol': 'green', 'Medium Vol': 'orange', 
                           'High Vol': 'red', 'Trending Up': 'darkgreen', 'Trending Down': 'darkred', 'Ranging': 'gray'}
            
            for regime, color in regime_colors.items():
                regime_data = top_overall[top_overall['regime'] == regime]
                if len(regime_data) > 0:
                    ax1.scatter(regime_data['stop_pct'], regime_data['target_pct'], 
                              s=regime_data['sharpe_ratio']*10, c=color, alpha=0.6, label=regime)
            
            ax1.set_xlabel('Stop %')
            ax1.set_ylabel('Target %')
            ax1.set_title('Top Configurations by Regime (size = Sharpe)')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. Performance by volatility regime
            ax2 = plt.subplot(2, 3, 2)
            vol_performance = results_df[results_df['regime'].isin(['Low Vol', 'Medium Vol', 'High Vol'])]
            if len(vol_performance) > 0:
                vol_summary = vol_performance.groupby(['regime', 'stop_pct', 'target_pct'])['sharpe_ratio'].mean().reset_index()
                vol_summary['config'] = vol_summary['stop_pct'].astype(str) + '/' + vol_summary['target_pct'].astype(str)
                
                # Get top 3 configs per regime
                top_configs = []
                for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                    regime_data = vol_summary[vol_summary['regime'] == regime].nlargest(3, 'sharpe_ratio')
                    top_configs.extend(regime_data.to_dict('records'))
                
                top_configs_df = pd.DataFrame(top_configs)
                pivot = top_configs_df.pivot(index='config', columns='regime', values='sharpe_ratio')
                pivot.plot(kind='bar', ax=ax2)
                ax2.set_title('Top Stop/Target Configs by Volatility')
                ax2.set_xlabel('Stop/Target Configuration')
                ax2.set_ylabel('Sharpe Ratio')
                ax2.legend(title='Regime')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # 3. Performance by trend regime
            ax3 = plt.subplot(2, 3, 3)
            trend_performance = results_df[results_df['regime'].isin(['Trending Up', 'Trending Down', 'Ranging'])]
            if len(trend_performance) > 0:
                trend_summary = trend_performance.groupby(['regime', 'stop_pct', 'target_pct'])['sharpe_ratio'].mean().reset_index()
                trend_summary['config'] = trend_summary['stop_pct'].astype(str) + '/' + trend_summary['target_pct'].astype(str)
                
                # Get top 3 configs per regime
                top_configs = []
                for regime in ['Trending Up', 'Trending Down', 'Ranging']:
                    regime_data = trend_summary[trend_summary['regime'] == regime].nlargest(3, 'sharpe_ratio')
                    top_configs.extend(regime_data.to_dict('records'))
                
                top_configs_df = pd.DataFrame(top_configs)
                pivot = top_configs_df.pivot(index='config', columns='regime', values='sharpe_ratio')
                pivot.plot(kind='bar', ax=ax3)
                ax3.set_title('Top Stop/Target Configs by Trend')
                ax3.set_xlabel('Stop/Target Configuration')
                ax3.set_ylabel('Sharpe Ratio')
                ax3.legend(title='Regime')
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            # 4. Win rate comparison
            ax4 = plt.subplot(2, 3, 4)
            win_rate_data = results_df[results_df['regime'] == 'Overall'].copy()
            win_rate_data['config'] = win_rate_data['stop_pct'].astype(str) + '/' + win_rate_data['target_pct'].astype(str)
            top_win_configs = win_rate_data.nlargest(10, 'sharpe_ratio')
            
            x = range(len(top_win_configs))
            ax4.bar(x, top_win_configs['win_rate'] * 100)
            ax4.set_xticks(x)
            ax4.set_xticklabels(top_win_configs['config'], rotation=45)
            ax4.set_ylabel('Win Rate %')
            ax4.set_title('Win Rates of Top 10 Configurations')
            ax4.axhline(50, color='red', linestyle='--', alpha=0.5)
            
            # 5. Return distribution
            ax5 = plt.subplot(2, 3, 5)
            return_comparison = results_df[results_df['regime'] == 'Overall'].copy()
            return_comparison['improvement'] = (return_comparison['total_return'] - return_comparison['base_return']) * 100
            
            top_returns = return_comparison.nlargest(15, 'total_return')
            top_returns['config'] = top_returns['stop_pct'].astype(str) + '/' + top_returns['target_pct'].astype(str)
            
            ax5.scatter(top_returns['base_return'] * 100, top_returns['total_return'] * 100, 
                       s=100, alpha=0.6, c=top_returns['sharpe_ratio'], cmap='viridis')
            
            # Add labels for top 5
            for i, row in top_returns.head(5).iterrows():
                ax5.annotate(row['config'], (row['base_return']*100, row['total_return']*100), 
                           fontsize=8, alpha=0.7)
            
            ax5.plot([-50, 50], [-50, 50], 'r--', alpha=0.5)
            ax5.set_xlabel('Base Return %')
            ax5.set_ylabel('Modified Return %')
            ax5.set_title('Return Improvement (color = Sharpe)')
            ax5.grid(True, alpha=0.3)
            
            # 6. Summary recommendations
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            # Generate dynamic recommendations
            recommendations = "📋 RECOMMENDATIONS\n\n"
            
            # Find most consistent configuration
            config_performance = results_df.groupby(['stop_pct', 'target_pct']).agg({
                'sharpe_ratio': ['mean', 'std', 'count'],
                'win_rate': 'mean'
            }).reset_index()
            
            config_performance.columns = ['stop_pct', 'target_pct', 'avg_sharpe', 'sharpe_std', 'count', 'avg_win_rate']
            config_performance['consistency'] = config_performance['avg_sharpe'] / (config_performance['sharpe_std'] + 0.1)
            
            most_consistent = config_performance.nlargest(1, 'consistency').iloc[0]
            recommendations += f"1. Most Consistent:\n"
            recommendations += f"   {most_consistent['stop_pct']}%/{most_consistent['target_pct']}% stop/target\n"
            recommendations += f"   Avg Sharpe: {most_consistent['avg_sharpe']:.2f}\n\n"
            
            # Regime-specific recommendations
            recommendations += "2. Regime-Adaptive Strategy:\n"
            
            vol_recs = []
            for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                regime_best = results_df[results_df['regime'] == regime].nlargest(1, 'sharpe_ratio')
                if len(regime_best) > 0:
                    best = regime_best.iloc[0]
                    vol_recs.append(f"   {regime}: {best['stop_pct']}/{best['target_pct']}%")
            
            recommendations += "\n".join(vol_recs) + "\n\n"
            
            # Trend recommendations
            recommendations += "3. Trend-Based Adjustments:\n"
            uptrend_configs = results_df[results_df['regime'] == 'Trending Up'].nlargest(3, 'sharpe_ratio')
            if len(uptrend_configs) > 0:
                avg_ratio = uptrend_configs['target_pct'].mean() / uptrend_configs['stop_pct'].mean()
                recommendations += f"   Uptrend: Use {avg_ratio:.1f}:1 reward/risk\n"
            
            downtrend_configs = results_df[results_df['regime'] == 'Trending Down'].nlargest(3, 'sharpe_ratio')
            if len(downtrend_configs) > 0:
                avg_ratio = downtrend_configs['target_pct'].mean() / downtrend_configs['stop_pct'].mean()
                recommendations += f"   Downtrend: Use {avg_ratio:.1f}:1 reward/risk\n"
            
            # 1-minute specific insights
            recommendations += "\n4. 1-Minute Specific Insights:\n"
            avg_stop = overall_best['stop_pct'].mean()
            avg_target = overall_best['target_pct'].mean()
            recommendations += f"   Optimal range: {avg_stop:.3f}-{avg_target:.3f}%\n"
            recommendations += f"   Tighter than 5m data suggests\n"
            
            ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes,
                    verticalalignment='top', fontsize=11, family='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))
            
            plt.tight_layout()
            plt.show()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df.to_csv(run_dir / f'regime_analysis_1m_{timestamp}.csv', index=False)
            
            # Create regime configuration file
            regime_config = {}
            for regime in ['Low Vol', 'Medium Vol', 'High Vol', 'Trending Up', 'Trending Down', 'Ranging']:
                regime_best = results_df[results_df['regime'] == regime].nlargest(1, 'sharpe_ratio')
                if len(regime_best) > 0:
                    best = regime_best.iloc[0]
                    regime_config[regime] = {
                        'stop_pct': float(best['stop_pct']),
                        'target_pct': float(best['target_pct']),
                        'expected_sharpe': float(best['sharpe_ratio']),
                        'expected_return': float(best['total_return'])
                    }
            
            # Save regime config
            import json
            with open(run_dir / f'optimal_regime_config_{timestamp}.json', 'w') as f:
                json.dump(regime_config, f, indent=2)
            
            print(f"\n✅ Analysis complete! Results saved to:")
            print(f"  - {run_dir}/regime_analysis_1m_{timestamp}.csv")
            print(f"  - {run_dir}/optimal_regime_config_{timestamp}.json")
            
else:
    print("❌ No performance data found. Please run backtest first.")

print("\n" + "="*80)
print("💡 Next Steps:")
print("1. Review regime-specific configurations")
print("2. Implement adaptive stop/target based on current regime")
print("3. Consider tighter stops for 1-minute vs 5-minute data")
print("4. Test on out-of-sample data with regime adaptation")