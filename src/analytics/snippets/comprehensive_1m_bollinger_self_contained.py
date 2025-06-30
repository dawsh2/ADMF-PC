# Comprehensive 1-Minute Bollinger Analysis - Self Contained
# This version includes all necessary functions and doesn't rely on notebook context

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
ANALYZE_ALL_STRATEGIES = True
TOP_N_STRATEGIES = 20
execution_cost_bps = 1.0  # Default execution cost

# Set default run_dir if not already defined
if 'run_dir' not in globals():
    run_dir = Path('config/bollinger/results/latest')
    print(f"Using default run_dir: {run_dir}")

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

# Regime-specific configurations
REGIME_CONFIGS = {
    'Low Vol': [(0.03, 0.05), (0.05, 0.075), (0.05, 0.10)],
    'Medium Vol': [(0.05, 0.10), (0.075, 0.10), (0.075, 0.15)],
    'High Vol': [(0.075, 0.15), (0.10, 0.15), (0.10, 0.20)],
    'Trending Up': [(0.05, 0.15), (0.075, 0.20), (0.10, 0.30)],
    'Trending Down': [(0.10, 0.075), (0.15, 0.10), (0.20, 0.15)],
    'Ranging': [(0.05, 0.05), (0.075, 0.075), (0.10, 0.10)]
}

def extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """Extract individual trades from strategy signals with proper timezone handling."""
    # Load trace file
    try:
        trace_df = pd.read_parquet(trace_path)
    except Exception as e:
        print(f"Error reading trace file {trace_path}: {e}")
        return pd.DataFrame()
    
    # Filter for this strategy's signals
    if 'strategy_hash' in trace_df.columns:
        strategy_signals = trace_df[trace_df['strategy_hash'] == strategy_hash].copy()
    else:
        # If no strategy_hash column, assume all signals are for this strategy
        strategy_signals = trace_df.copy()
    
    if len(strategy_signals) == 0:
        return pd.DataFrame()
    
    strategy_signals = strategy_signals.sort_values('ts')
    
    # Work with a copy to avoid modifying the original
    market_data_copy = market_data.copy()
    
    # Ensure both timestamps are timezone-naive datetime64[ns]
    # Convert signal timestamps (they come as strings from parquet)
    strategy_signals['ts'] = pd.to_datetime(strategy_signals['ts'])
    if hasattr(strategy_signals['ts'].dtype, 'tz'):
        strategy_signals['ts'] = strategy_signals['ts'].dt.tz_localize(None)
    
    # Ensure market data timestamp is also timezone-naive
    if not pd.api.types.is_datetime64_any_dtype(market_data_copy['timestamp']):
        market_data_copy['timestamp'] = pd.to_datetime(market_data_copy['timestamp'])
    if hasattr(market_data_copy['timestamp'].dtype, 'tz') and market_data_copy['timestamp'].dt.tz is not None:
        market_data_copy['timestamp'] = market_data_copy['timestamp'].dt.tz_localize(None)
    
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
            market_idx = closest_idx
        else:
            continue
        
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
                
                if direction == 1:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                execution_cost = execution_cost_bps / 10000
                net_return = raw_return - execution_cost * 2
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'direction': direction,
                    'raw_return': raw_return,
                    'execution_cost': execution_cost * 2,
                    'net_return': net_return,
                    'duration_bars': exit_idx - entry_idx
                })
            
            current_position = signal_value
            if signal_value != 0:
                entry_idx = market_idx
                entry_price = current_price
                entry_time = timestamp
            else:
                entry_idx = None
    
    return pd.DataFrame(trades)

def apply_stop_target(trades_df, stop_pct, target_pct, market_data):
    """Apply stop loss and profit target to trades"""
    if len(trades_df) == 0:
        return np.array([]), {'stop': 0, 'target': 0, 'signal': 0}
        
    if stop_pct == 0 and target_pct == 0:
        return trades_df['net_return'].values, {'stop': 0, 'target': 0, 'signal': len(trades_df)}
    
    modified_returns = []
    exit_types = {'stop': 0, 'target': 0, 'signal': 0}
    
    # Handle both 'close' and 'Close' column names
    close_col = 'Close' if 'Close' in market_data.columns else 'close'
    low_col = 'Low' if 'Low' in market_data.columns else 'low'
    high_col = 'High' if 'High' in market_data.columns else 'high'
    
    for _, trade in trades_df.iterrows():
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) == 0:
            modified_returns.append(trade['net_return'])
            exit_types['signal'] += 1
            continue
        
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Set stop and target prices
        if target_pct > 0:
            if direction == 1:  # Long
                stop_price = entry_price * (1 - stop_pct/100) if stop_pct > 0 else 0
                target_price = entry_price * (1 + target_pct/100)
            else:  # Short
                stop_price = entry_price * (1 + stop_pct/100) if stop_pct > 0 else float('inf')
                target_price = entry_price * (1 - target_pct/100)
        else:
            if direction == 1:
                stop_price = entry_price * (1 - stop_pct/100)
                target_price = float('inf')
            else:
                stop_price = entry_price * (1 + stop_pct/100)
                target_price = 0
        
        # Check each bar for exit
        exit_price = trade['exit_price']
        exit_type = 'signal'
        
        for _, bar in trade_prices.iterrows():
            if direction == 1:  # Long
                if stop_pct > 0 and bar[low_col] <= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    break
                elif target_pct > 0 and bar[high_col] >= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    break
            else:  # Short
                if stop_pct > 0 and bar[high_col] >= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    break
                elif target_pct > 0 and bar[low_col] <= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    break
        
        exit_types[exit_type] += 1
        
        # Calculate return
        if direction == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price
        
        net_return = raw_return - trade['execution_cost']
        modified_returns.append(net_return)
    
    return np.array(modified_returns), exit_types

def calculate_performance_metrics(returns_array, num_trades, trading_days=252):
    """Calculate performance metrics from returns array"""
    if len(returns_array) == 0:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'avg_return': 0,
            'max_drawdown': 0
        }
    
    total_return = (1 + returns_array).prod() - 1
    win_rate = (returns_array > 0).mean()
    avg_return = returns_array.mean()
    
    if returns_array.std() > 0:
        sharpe_ratio = returns_array.mean() / returns_array.std() * np.sqrt(252 * num_trades / max(trading_days, 1))
    else:
        sharpe_ratio = 0
    
    # Calculate max drawdown
    cumulative = (1 + returns_array).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'max_drawdown': max_drawdown
    }

def get_actual_trading_days(performance_df, market_data):
    """Calculate actual trading days based on market data"""
    trading_days = len(market_data['timestamp'].dt.date.unique())
    date_range = f"{market_data['timestamp'].min().date()} to {market_data['timestamp'].max().date()}"
    print(f"Actual trading period: {date_range} ({trading_days} days)")
    return trading_days

def calculate_market_regimes(market_data):
    """Calculate volatility and trend regimes"""
    # Determine the close column name
    close_col = 'Close' if 'Close' in market_data.columns else 'close'
    
    # Volatility regime (20-period rolling)
    market_data['returns'] = market_data[close_col].pct_change()
    market_data['volatility'] = market_data['returns'].rolling(window=20*60).std() * np.sqrt(252*390)
    
    vol_percentiles = market_data['volatility'].quantile([0.33, 0.67])
    market_data['vol_regime'] = pd.cut(
        market_data['volatility'],
        bins=[0, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
        labels=['Low Vol', 'Medium Vol', 'High Vol']
    )
    
    # Trend regime
    market_data['sma_60'] = market_data[close_col].rolling(window=60).mean()
    market_data['sma_240'] = market_data[close_col].rolling(window=240).mean()
    
    market_data['trend_strength'] = (market_data[close_col] - market_data['sma_240']) / market_data['sma_240'] * 100
    
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
                if len(regime_trades) > 5:  # Need minimum trades
                    regime_returns, _ = apply_stop_target(regime_trades, stop_pct, target_pct, market_data)
                    if len(regime_returns) > 0:
                        regime_stats = calculate_performance_metrics(regime_returns, len(regime_trades))
                        regime_stats['stop_pct'] = stop_pct
                        regime_stats['target_pct'] = target_pct
                        regime_stats['regime'] = vol_regime
                        results.append(regime_stats)
    
    return pd.DataFrame(results)

# Main analysis
print("📊 Comprehensive 1-Minute Bollinger Analysis")
print("=" * 80)

# Check prerequisites
if 'market_data' not in globals() or 'performance_df' not in globals():
    print("❌ Missing required data!")
    print("Please ensure you have:")
    print("1. market_data - your market data DataFrame")
    print("2. performance_df - your performance metrics DataFrame")
    print("3. run_dir - the directory containing results")
else:
    # Clean market_data timestamps if needed
    if 'timestamp' in market_data.columns:
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
    
    # Calculate market regimes
    print("Calculating market regimes...")
    market_data = calculate_market_regimes(market_data)
    
    # Get actual trading days
    actual_trading_days = get_actual_trading_days(performance_df, market_data)
    
    # Basic statistics
    print(f"\nTotal strategies: {len(performance_df)}")
    
    # Filter strategies with trades
    strategies_with_trades = performance_df[performance_df['num_trades'] > 0]
    print(f"Strategies with trades: {len(strategies_with_trades)}")
    
    if len(strategies_with_trades) == 0:
        print("\n⚠️ No strategies have trades in the performance data!")
        print("Checking signal files directly...")
        
        # Try to analyze signals directly
        if 'run_dir' in globals():
            signal_dir = Path(run_dir) / 'traces' / 'signals' / 'bollinger_bands'
            if signal_dir.exists():
                parquet_files = list(signal_dir.glob('*.parquet'))
                print(f"Found {len(parquet_files)} signal files")
                
                # Sample a few files to check for signals
                sample_files = parquet_files[:5]
                for f in sample_files:
                    df = pd.read_parquet(f)
                    non_zero = (df['val'] != 0).sum()
                    print(f"  {f.name}: {len(df)} rows, {non_zero} non-zero signals")
    else:
        # Determine strategies to analyze
        if ANALYZE_ALL_STRATEGIES:
            strategies_to_analyze = strategies_with_trades
            print(f"\nAnalyzing ALL {len(strategies_to_analyze)} strategies (this may take a while)...")
        else:
            strategies_to_analyze = strategies_with_trades.nlargest(TOP_N_STRATEGIES, 'num_trades')
            print(f"\nAnalyzing top {len(strategies_to_analyze)} strategies by trade count")
        
        all_results = []
        
        for idx, (_, strategy) in enumerate(strategies_to_analyze.iterrows()):
            print(f"\rProcessing strategy {idx+1}/{len(strategies_to_analyze)}...", end='', flush=True)
            
            # Ensure full path to trace file
            if 'run_dir' in globals():
                full_trace_path = Path(run_dir) / strategy['trace_path']
            else:
                full_trace_path = strategy['trace_path']
            
            try:
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
                    all_results.append(regime_results)
            except Exception as e:
                print(f"\nError processing strategy {strategy['strategy_hash']}: {e}")
        
        print("\n\nProcessing complete!")
        
        if all_results:
            # Combine all results
            results_df = pd.concat(all_results, ignore_index=True)
            
            # Find best overall configuration
            print("\n" + "=" * 80)
            print("📈 OPTIMAL STOP/TARGET CONFIGURATIONS")
            print("=" * 80)
            
            # Best overall
            best_overall = results_df[results_df['regime'] == 'Overall'].nlargest(5, 'sharpe_ratio')
            print("\nTop 5 configurations (Overall):")
            for idx, row in best_overall.iterrows():
                print(f"  Stop: {row['stop_pct']}%, Target: {row['target_pct']}% → Sharpe: {row['sharpe_ratio']:.2f}")
            
            # Best by regime
            for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                regime_data = results_df[results_df['regime'] == regime]
                if len(regime_data) > 0:
                    best_regime = regime_data.nlargest(3, 'sharpe_ratio')
                    print(f"\nTop configurations for {regime}:")
                    for idx, row in best_regime.iterrows():
                        print(f"  Stop: {row['stop_pct']}%, Target: {row['target_pct']}% → Sharpe: {row['sharpe_ratio']:.2f}")
            
            # Save results
            if 'run_dir' in globals():
                results_path = Path(run_dir) / 'stop_target_regime_analysis.csv'
                results_df.to_csv(results_path, index=False)
                print(f"\n✅ Results saved to: {results_path}")
        else:
            print("\n❌ No results generated. Check that strategies have trades.")

print("\n" + "=" * 80)
print("💡 Analysis complete!")
print("=" * 80)