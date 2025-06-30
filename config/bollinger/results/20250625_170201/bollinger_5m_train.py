#!/usr/bin/env python
# coding: utf-8

# # Universal Strategy Analysis
# 
# This notebook provides comprehensive analysis across all strategies tested in a parameter sweep.
# 
# **Key Features:**
# - Cross-strategy performance comparison
# - Parameter sensitivity analysis
# - Correlation analysis for ensemble building
# - Regime-specific performance breakdown
# - Automatic identification of optimal strategies and ensembles

# In[20]:


# Parameters will be injected here by papermill
# This cell is tagged with 'parameters' for papermill to recognize it
run_dir = "."
config_name = "config"
symbols = ["SPY"]
timeframe = "5m"
min_strategies_to_analyze = 20
sharpe_threshold = 1.0
correlation_threshold = 0.7
top_n_strategies = 10
ensemble_size = 5
calculate_all_performance = True  # Set to False to limit analysis for large sweeps
performance_limit = 100  # If calculate_all_performance is False, limit to this many

# Enhanced analysis parameters
execution_cost_bps = 1.0  # Round-trip execution cost in basis points
analyze_stop_losses = True  # Whether to analyze stop loss impact
stop_loss_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.75, 1.0]  # Stop loss percentages
verify_intraday = True  # Whether to verify intraday constraints
market_timezone = "America/New_York"  # Market timezone for constraint verification


# In[21]:


# Parameters
run_dir = "/Users/daws/ADMF-PC/config/bollinger/results/20250625_170201"
config_name = "bollinger"
symbols = ["SPY"]
timeframe = "5m"
min_strategies_to_analyze = 20
sharpe_threshold = 1.0
correlation_threshold = 0.7
top_n_strategies = 10
ensemble_size = 5
calculate_all_performance = True
performance_limit = 100


# In[22]:


# Delete performance cache to allow recalculation with different execution costs
# Uncomment the next line to force recalculation
# !rm -f performance_metrics.parquet

# Alternative: Set this to True to ignore cache and always recalculate
IGNORE_CACHE = False  # Set to True when testing different execution costs


# ## Setup

# In[23]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import duckdb
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Initialize DuckDB
con = duckdb.connect()

# Convert run_dir to Path and resolve to absolute path
run_dir = Path(run_dir).resolve()
print(f"Analyzing run: {run_dir.name}")
print(f"Full path: {run_dir}")
print(f"Config: {config_name}")
print(f"Symbol(s): {symbols}")
print(f"Timeframe: {timeframe}")


# In[24]:


# Enhanced analysis helper functions
import pytz
from datetime import time

def extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Extract trades from signal trace with execution costs.

    Args:
        strategy_hash: Strategy identifier
        trace_path: Path to trace file
        market_data: Market price data
        execution_cost_bps: Round-trip execution cost in basis points (default 1bp)

    Returns:
        DataFrame with trade details including costs
    """
    try:
        signals_path = run_dir / trace_path
        signals = pd.read_parquet(signals_path)
        signals['ts'] = pd.to_datetime(signals['ts'])

        # Merge with market data
        df = market_data.merge(
            signals[['ts', 'val', 'px']], 
            left_on='timestamp', 
            right_on='ts', 
            how='left'
        )

        # Forward fill signals
        df['signal'] = df['val'].ffill().fillna(0)
        df['position'] = df['signal'].replace({0: 0, 1: 1, -1: -1})
        df['position_change'] = df['position'].diff().fillna(0)

        trades = []
        current_trade = None

        for idx, row in df.iterrows():
            if row['position_change'] != 0 and row['position'] != 0:
                # New position opened
                if current_trade is None:
                    current_trade = {
                        'entry_time': row['timestamp'],
                        'entry_price': row['px'] if pd.notna(row['px']) else row['close'],
                        'direction': row['position'],
                        'entry_idx': idx
                    }
            elif current_trade is not None and (row['position'] == 0 or row['position_change'] != 0):
                # Position closed
                exit_price = row['px'] if pd.notna(row['px']) else row['close']

                # Avoid division by zero - check if entry price is valid
                if current_trade['entry_price'] == 0 or pd.isna(current_trade['entry_price']):
                    print(f"Warning: Invalid entry price {current_trade['entry_price']} for trade at {current_trade['entry_time']}")
                    current_trade = None
                    continue

                # Calculate raw return
                if current_trade['direction'] == 1:  # Long
                    raw_return = (exit_price - current_trade['entry_price']) / current_trade['entry_price']
                else:  # Short
                    raw_return = (current_trade['entry_price'] - exit_price) / current_trade['entry_price']

                # Apply execution costs
                cost_adjustment = execution_cost_bps / 10000  # Convert bps to decimal
                net_return = raw_return - cost_adjustment

                trade = {
                    'strategy_hash': strategy_hash,
                    'entry_time': current_trade['entry_time'],
                    'exit_time': row['timestamp'],
                    'entry_price': current_trade['entry_price'],
                    'exit_price': exit_price,
                    'direction': current_trade['direction'],
                    'raw_return': raw_return,
                    'execution_cost': cost_adjustment,
                    'net_return': net_return,
                    'duration_minutes': (row['timestamp'] - current_trade['entry_time']).total_seconds() / 60,
                    'entry_idx': current_trade['entry_idx'],
                    'exit_idx': idx
                }
                trades.append(trade)

                # Reset for next trade
                current_trade = None
                if row['position'] != 0 and row['position_change'] != 0:
                    # Immediately open new position (reversal)
                    current_trade = {
                        'entry_time': row['timestamp'],
                        'entry_price': row['px'] if pd.notna(row['px']) else row['close'],
                        'direction': row['position'],
                        'entry_idx': idx
                    }

        return pd.DataFrame(trades)
    except Exception as e:
        print(f"Error extracting trades for {strategy_hash[:8]}: {e}")
        return pd.DataFrame()

def calculate_stop_loss_impact(trades_df, stop_loss_levels=None, market_data=None):
    """
    Calculate returns with various stop loss levels using proper intraday simulation.

    Args:
        trades_df: DataFrame of trades (must include entry_idx and exit_idx)
        stop_loss_levels: List of stop loss percentages (default 0.05% to 1%)
        market_data: Market data for intraday price movements

    Returns:
        DataFrame with returns for each stop loss level
    """
    if stop_loss_levels is None:
        stop_loss_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]

    if market_data is None:
        print("WARNING: No market data provided for stop loss analysis. Using simplified calculation.")
        # Fall back to the flawed method if no market data
        return calculate_stop_loss_impact_simple(trades_df, stop_loss_levels)

    results = []

    for sl_pct in stop_loss_levels:
        sl_decimal = sl_pct / 100

        trades_with_sl = []
        stopped_out_count = 0

        # Process each trade with intraday stop loss
        for _, trade in trades_df.iterrows():
            # Get intraday prices for this trade
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]

            if len(trade_prices) == 0:
                continue

            entry_price = trade['entry_price']
            direction = trade['direction']

            # Calculate stop loss price
            if direction == 1:  # Long position
                stop_price = entry_price * (1 - sl_decimal)
            else:  # Short position  
                stop_price = entry_price * (1 + sl_decimal)

            # Check if stop loss is hit
            stopped = False
            exit_price = trade['exit_price']
            exit_time = trade['exit_time']

            for idx, bar in trade_prices.iterrows():
                if direction == 1:  # Long
                    # Check if low price hits stop
                    if bar['low'] <= stop_price:
                        stopped = True
                        stopped_out_count += 1
                        exit_price = stop_price
                        exit_time = bar['timestamp']
                        break
                else:  # Short
                    # Check if high price hits stop
                    if bar['high'] >= stop_price:
                        stopped = True
                        stopped_out_count += 1
                        exit_price = stop_price
                        exit_time = bar['timestamp']
                        break

            # Calculate return with actual or stopped exit
            if direction == 1:  # Long
                raw_return = (exit_price - entry_price) / entry_price
            else:  # Short
                raw_return = (entry_price - exit_price) / entry_price

            # Apply execution costs
            net_return = raw_return - trade['execution_cost']

            trade_result = trade.copy()
            trade_result['raw_return'] = raw_return
            trade_result['net_return'] = net_return
            trade_result['stopped_out'] = stopped
            if stopped:
                trade_result['exit_price'] = exit_price
                trade_result['exit_time'] = exit_time

            trades_with_sl.append(trade_result)

        trades_with_sl_df = pd.DataFrame(trades_with_sl)

        if len(trades_with_sl_df) > 0:
            # Calculate metrics with stop loss
            total_return = trades_with_sl_df['net_return'].sum()
            avg_return = trades_with_sl_df['net_return'].mean()
            win_rate = (trades_with_sl_df['net_return'] > 0).mean()

            results.append({
                'stop_loss_pct': sl_pct,
                'total_return': total_return,
                'avg_return_per_trade': avg_return,
                'win_rate': win_rate,
                'stopped_out_count': stopped_out_count,
                'stopped_out_rate': stopped_out_count / len(trades_with_sl_df),
                'num_trades': len(trades_with_sl_df),
                'avg_winner': trades_with_sl_df[trades_with_sl_df['net_return'] > 0]['net_return'].mean() if (trades_with_sl_df['net_return'] > 0).any() else 0,
                'avg_loser': trades_with_sl_df[trades_with_sl_df['net_return'] <= 0]['net_return'].mean() if (trades_with_sl_df['net_return'] <= 0).any() else 0
            })

    return pd.DataFrame(results)

def calculate_stop_loss_impact_simple(trades_df, stop_loss_levels):
    """
    Simplified (flawed) stop loss calculation - only caps losses retrospectively.
    Kept for comparison purposes.
    """
    results = []

    for sl_pct in stop_loss_levels:
        sl_decimal = sl_pct / 100

        trades_with_sl = trades_df.copy()
        stopped_out_count = 0

        # Apply stop loss to each trade (FLAWED - only caps losses)
        for idx, trade in trades_with_sl.iterrows():
            if trade['raw_return'] < -sl_decimal:
                trades_with_sl.loc[idx, 'raw_return'] = -sl_decimal
                trades_with_sl.loc[idx, 'net_return'] = -sl_decimal - trade['execution_cost']
                stopped_out_count += 1

        # Calculate metrics with stop loss
        total_return = trades_with_sl['net_return'].sum()
        avg_return = trades_with_sl['net_return'].mean()
        win_rate = (trades_with_sl['net_return'] > 0).mean()

        results.append({
            'stop_loss_pct': sl_pct,
            'total_return': total_return,
            'avg_return_per_trade': avg_return,
            'win_rate': win_rate,
            'stopped_out_count': stopped_out_count,
            'stopped_out_rate': stopped_out_count / len(trades_with_sl) if len(trades_with_sl) > 0 else 0,
            'num_trades': len(trades_with_sl)
        })

    return pd.DataFrame(results)

def verify_intraday_constraint(trades_df, market_tz='America/New_York'):
    """
    Verify that trades respect intraday constraints.

    Args:
        trades_df: DataFrame of trades
        market_tz: Market timezone (default NYSE)

    Returns:
        Dictionary with constraint verification results
    """
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'overnight_positions': 0,
            'overnight_position_pct': 0,
            'after_hours_entries': 0,
            'after_hours_exits': 0,
            'fully_intraday': 0,
            'avg_trade_duration_minutes': 0,
            'max_trade_duration_minutes': 0,
            'trades_over_390_minutes': 0
        }

    # Convert to market timezone
    market_tz_obj = pytz.timezone(market_tz)

    trades_df = trades_df.copy()

    # Handle timezone conversion properly
    # First check if timestamps are already timezone-aware
    entry_times = pd.to_datetime(trades_df['entry_time'])
    exit_times = pd.to_datetime(trades_df['exit_time'])

    if entry_times.dt.tz is not None:
        # Already timezone-aware, just convert
        trades_df['entry_time_mkt'] = entry_times.dt.tz_convert(market_tz_obj)
    else:
        # Timezone-naive, localize first then convert
        trades_df['entry_time_mkt'] = entry_times.dt.tz_localize('UTC').dt.tz_convert(market_tz_obj)

    if exit_times.dt.tz is not None:
        # Already timezone-aware, just convert
        trades_df['exit_time_mkt'] = exit_times.dt.tz_convert(market_tz_obj)
    else:
        # Timezone-naive, localize first then convert
        trades_df['exit_time_mkt'] = exit_times.dt.tz_localize('UTC').dt.tz_convert(market_tz_obj)

    # Market hours (9:30 AM - 4:00 PM ET)
    market_open = time(9, 30)
    market_close = time(16, 0)

    # Check for overnight positions
    trades_df['entry_date'] = trades_df['entry_time_mkt'].dt.date
    trades_df['exit_date'] = trades_df['exit_time_mkt'].dt.date
    trades_df['overnight'] = trades_df['entry_date'] != trades_df['exit_date']

    # Check for after-hours trades
    trades_df['entry_time_only'] = trades_df['entry_time_mkt'].dt.time
    trades_df['exit_time_only'] = trades_df['exit_time_mkt'].dt.time

    trades_df['after_hours_entry'] = (
        (trades_df['entry_time_only'] < market_open) | 
        (trades_df['entry_time_only'] >= market_close)
    )
    trades_df['after_hours_exit'] = (
        (trades_df['exit_time_only'] < market_open) | 
        (trades_df['exit_time_only'] >= market_close)
    )

    results = {
        'total_trades': len(trades_df),
        'overnight_positions': trades_df['overnight'].sum(),
        'overnight_position_pct': trades_df['overnight'].mean() * 100,
        'after_hours_entries': trades_df['after_hours_entry'].sum(),
        'after_hours_exits': trades_df['after_hours_exit'].sum(),
        'fully_intraday': (~trades_df['overnight']).sum(),
        'avg_trade_duration_minutes': trades_df['duration_minutes'].mean(),
        'max_trade_duration_minutes': trades_df['duration_minutes'].max(),
        'trades_over_390_minutes': (trades_df['duration_minutes'] > 390).sum()  # Full trading day
    }

    # Add hourly breakdown
    trades_df['entry_hour'] = trades_df['entry_time_mkt'].dt.hour
    trades_df['exit_hour'] = trades_df['exit_time_mkt'].dt.hour

    results['entries_by_hour'] = trades_df['entry_hour'].value_counts().to_dict()
    results['exits_by_hour'] = trades_df['exit_hour'].value_counts().to_dict()

    return results


# In[25]:


# Setup path for loading analysis snippets
import sys
from pathlib import Path

# Find the project root (where src/ directory is)
current_path = Path(run_dir).resolve()
project_root = None

# Search up the directory tree for src/analytics/snippets
for parent in current_path.parents:
    if (parent / 'src' / 'analytics' / 'snippets').exists():
        project_root = parent
        break

# If not found from run_dir, try from current working directory
if not project_root:
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / 'src' / 'analytics' / 'snippets').exists():
            project_root = parent
            break

# Last resort: check common project locations
if not project_root:
    common_roots = [
        Path('/Users/daws/ADMF-PC'),
        Path.home() / 'ADMF-PC',
        Path.cwd().parent.parent.parent.parent  # 4 levels up from typical results dir
    ]
    for root in common_roots:
        if root.exists() and (root / 'src' / 'analytics' / 'snippets').exists():
            project_root = root
            break

if project_root:
    # Add to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    snippets_path = project_root / 'src' / 'analytics' / 'snippets'
    queries_path = project_root / 'src' / 'analytics' / 'queries'
    print(f"✅ Found project root: {project_root}")
    print(f"✅ Analysis snippets available at: {snippets_path}")
    print(f"✅ SQL queries available at: {queries_path}")
    print("\nUse %load to load any snippet, e.g.:")
    print("  %load {}/src/analytics/snippets/exploratory/signal_frequency.py".format(project_root))
    print("  %load {}/src/analytics/snippets/ensembles/find_uncorrelated.py".format(project_root))
else:
    print("⚠️ Could not find project root with src/analytics/snippets")
    print(f"  Searched from: {current_path}")
    print(f"  Current working directory: {Path.cwd()}")


# ## Load Strategy Index

# In[26]:


# Load strategy index - the catalog of all strategies tested
strategy_index_path = run_dir / 'strategy_index.parquet'

if strategy_index_path.exists():
    strategy_index = pd.read_parquet(strategy_index_path)
    print(f"✅ Loaded {len(strategy_index)} strategies from {strategy_index_path}")

    # Show strategy type distribution
    by_type = strategy_index['strategy_type'].value_counts()
    print("\nStrategies by type:")
    for stype, count in by_type.items():
        print(f"  {stype}: {count}")

    # Show sample of columns
    print(f"\nColumns: {list(strategy_index.columns)[:10]}...")
else:
    print(f"❌ No strategy_index.parquet found at {strategy_index_path}")
    strategy_index = None


# In[27]:


# Load market data
market_data = None
for symbol in symbols:
    try:
        # Try different possible locations for market data
        data_paths = [
            run_dir / f'data/{symbol}_{timeframe}.csv',
            run_dir / f'{symbol}_{timeframe}.csv',
            run_dir.parent / f'data/{symbol}_{timeframe}.csv',
            Path(f'/Users/daws/ADMF-PC/data/{symbol}_{timeframe}.csv')
        ]

        for data_path in data_paths:
            if data_path.exists():
                market_data = pd.read_csv(data_path)
                market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
                market_data = market_data.sort_values('timestamp')
                print(f"✅ Loaded market data from: {data_path}")
                print(f"   Date range: {market_data['timestamp'].min()} to {market_data['timestamp'].max()}")
                print(f"   Total bars: {len(market_data)}")
                break

        if market_data is not None:
            break

    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")

if market_data is None:
    print("❌ Could not load market data")
    print("Tried paths:")
    for path in data_paths:
        print(f"  - {path}")


# In[28]:


def calculate_performance(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """Calculate performance metrics using TRADE-BASED approach for consistency"""
    try:
        # Extract actual trades
        trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)

        if len(trades) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_return_per_trade': 0,
                'profit_factor': 0,
                'total_execution_cost': 0
            }

        # Calculate cumulative returns from trades
        trades = trades.sort_values('entry_time').reset_index(drop=True)
        trades['cum_return'] = (1 + trades['net_return']).cumprod()
        total_return = trades['cum_return'].iloc[-1] - 1

        # Calculate Sharpe ratio from trade returns
        if trades['net_return'].std() > 0:
            # Annualize based on average trades per day
            days_in_data = (trades['exit_time'].max() - trades['entry_time'].min()).days
            if days_in_data > 0:
                trades_per_day = len(trades) / days_in_data
                annualization_factor = np.sqrt(252 * trades_per_day)
            else:
                annualization_factor = np.sqrt(252)
            sharpe = trades['net_return'].mean() / trades['net_return'].std() * annualization_factor
        else:
            sharpe = 0

        # Max drawdown from trade equity curve
        cummax = trades['cum_return'].expanding().max()
        drawdown = (trades['cum_return'] / cummax - 1)
        max_dd = drawdown.min()

        # Win rate and profit factor (trade-based)
        winning_trades = trades[trades['net_return'] > 0]
        losing_trades = trades[trades['net_return'] <= 0]

        win_rate = len(winning_trades) / len(trades)

        if len(losing_trades) > 0 and losing_trades['net_return'].sum() != 0:
            profit_factor = winning_trades['net_return'].sum() / abs(losing_trades['net_return'].sum())
        else:
            profit_factor = 999.99 if len(winning_trades) > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_return_per_trade': trades['net_return'].mean(),
            'profit_factor': profit_factor,
            'avg_winner': winning_trades['net_return'].mean() if len(winning_trades) > 0 else 0,
            'avg_loser': losing_trades['net_return'].mean() if len(losing_trades) > 0 else 0,
            'total_execution_cost': trades['execution_cost'].sum()
        }
    except Exception as e:
        print(f"Error calculating performance for {strategy_hash}: {e}")
        return None


# In[29]:


# Calculate performance for all strategies with execution costs
# NOTE: Now using TRADE-BASED metrics for all calculations (fixed inconsistency with stop loss analysis)
if strategy_index is not None and market_data is not None:
    performance_results = []

    # Determine strategies to analyze based on parameters
    strategies_to_analyze = strategy_index

    if not calculate_all_performance and len(strategy_index) > performance_limit:
        print(f"Note: Large parameter sweep detected ({len(strategy_index)} strategies)")
        print(f"Limiting analysis to {performance_limit} strategies (set calculate_all_performance=True to analyze all)")

        # Sample diverse strategies across all types
        strategies_to_analyze = strategy_index.groupby('strategy_type').apply(
            lambda x: x.sample(n=min(len(x), performance_limit // strategy_index['strategy_type'].nunique()), 
                             random_state=42)
        ).reset_index(drop=True)

    print(f"\nCalculating performance for {len(strategies_to_analyze)} strategies...")
    print(f"Using run directory: {run_dir}")
    print(f"Execution cost: {execution_cost_bps} basis points round-trip")
    print(f"📊 Using TRADE-BASED metrics (win rate = winning trades / total trades)")

    # Check if we already have cached performance metrics
    cached_performance_path = run_dir / 'performance_metrics.parquet'
    cache_metadata_path = run_dir / 'performance_cache_metadata.json'

    use_cache = False
    if cached_performance_path.exists() and cache_metadata_path.exists() and not IGNORE_CACHE:
        # Check if cache was created with same execution cost
        try:
            with open(cache_metadata_path, 'r') as f:
                cache_metadata = json.load(f)
            if cache_metadata.get('execution_cost_bps') == execution_cost_bps and cache_metadata.get('calculation_method') == 'trade_based':
                print(f"📂 Found cached trade-based performance metrics with same execution cost, loading...")
                performance_df = pd.read_parquet(cached_performance_path)
                print(f"✅ Loaded performance for {len(performance_df)} strategies from cache")
                use_cache = True
        except:
            pass

    if not use_cache:
        # Calculate performance
        for idx, row in strategies_to_analyze.iterrows():
            if idx % 10 == 0:
                print(f"  Progress: {idx}/{len(strategies_to_analyze)} ({idx/len(strategies_to_analyze)*100:.1f}%)")

            perf = calculate_performance(row['strategy_hash'], row['trace_path'], market_data, execution_cost_bps)

            if perf:
                # Combine strategy info with performance
                result = {**row.to_dict(), **perf}
                performance_results.append(result)

        print(f"  Progress: {len(strategies_to_analyze)}/{len(strategies_to_analyze)} (100.0%)")

        performance_df = pd.DataFrame(performance_results)
        print(f"\n✅ Calculated performance for {len(performance_df)} strategies")

        # Save performance results for future use (only if we calculated all)
        if calculate_all_performance and len(performance_df) == len(strategy_index):
            performance_df.to_parquet(cached_performance_path)
            # Save metadata about cache
            with open(cache_metadata_path, 'w') as f:
                json.dump({
                    'execution_cost_bps': execution_cost_bps,
                    'calculation_method': 'trade_based',
                    'created_at': datetime.now().isoformat()
                }, f)
            print(f"💾 Saved trade-based performance metrics to: {cached_performance_path}")
else:
    performance_df = pd.DataFrame()
    print("⚠️ Skipping performance calculation")


# ## Cross-Strategy Performance Analysis

# In[30]:


if len(performance_df) > 0:
    # Top performers across ALL strategy types
    top_overall = performance_df.nlargest(top_n_strategies, 'sharpe_ratio')

    print(f"\n🏆 Top {top_n_strategies} Strategies (All Types) - After {execution_cost_bps}bps Execution Costs:")
    print("=" * 90)

    # Look for parameter columns (both with and without param_ prefix for compatibility)
    all_param_cols = []
    # Check for param_ prefixed columns
    param_prefixed_cols = [col for col in top_overall.columns if col.startswith('param_')]
    # Check for direct parameter columns (per trace-updates.md)
    direct_param_cols = ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier', 'exit_threshold']
    available_param_cols = [col for col in direct_param_cols if col in top_overall.columns]

    # Use whichever we find
    if available_param_cols:
        all_param_cols = available_param_cols
    elif param_prefixed_cols:
        all_param_cols = param_prefixed_cols

    for idx, row in top_overall.iterrows():
        # Determine identifier to show
        strategy_identifier = row.get('strategy_id', 'unknown')
        if 'strategy_hash' in row and pd.notna(row['strategy_hash']):
            # Check if all strategies have the same hash
            if performance_df['strategy_hash'].nunique() > 1:
                # Use hash if they're unique
                strategy_identifier = row['strategy_hash'][:8]

        print(f"\n{row['strategy_type']} - {strategy_identifier}")
        print(f"  Sharpe: {row['sharpe_ratio']:.2f} | Return: {row['total_return']:.1%} | Drawdown: {row['max_drawdown']:.1%}")
        print(f"  Win Rate: {row['win_rate']:.1%} | Avg Return/Trade: {row['avg_return_per_trade']*100:.3f}% | Trades: {row['num_trades']}")

        # Show profit factor if available
        if 'profit_factor' in row:
            print(f"  Profit Factor: {row['profit_factor']:.2f} (Win$/Loss$)")

        # Show average winner/loser if available
        if 'avg_winner' in row and 'avg_loser' in row:
            print(f"  Avg Winner: +{row['avg_winner']*100:.2f}% | Avg Loser: {row['avg_loser']*100:.2f}%")

        print(f"  Total Execution Cost: {row['total_execution_cost']*100:.2f}%")

        # Show parameters
        if all_param_cols:
            # Filter out null parameters
            valid_params = []
            for col in all_param_cols[:5]:  # Show up to 5 parameters
                if col in row and pd.notna(row[col]):
                    param_name = col.replace('param_', '') if col.startswith('param_') else col
                    valid_params.append(f"{param_name}: {row[col]}")

            if valid_params:
                print(f"  Params: {' | '.join(valid_params)}")

    # Show profit factor distribution
    if 'profit_factor' in performance_df.columns:
        print("\n📈 Profit Factor Distribution (All Strategies):")
        pf_bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 999]
        pf_labels = ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '1.5-2.0', '>2.0']
        performance_df['pf_bin'] = pd.cut(performance_df['profit_factor'].clip(upper=100), bins=pf_bins, labels=pf_labels)
        print(performance_df['pf_bin'].value_counts().sort_index())

        # Show win rate distribution
        print("\n📊 Win Rate Distribution (Trade-Based):")
        print(f"Mean: {performance_df['win_rate'].mean():.1%}")
        print(f"Median: {performance_df['win_rate'].median():.1%}")
        print(f"Std Dev: {performance_df['win_rate'].std():.1%}")

        # Note about calculation method
        print("\n💡 Note: All metrics now use trade-based calculations:")
        print("• Win Rate = Winning Trades / Total Trades (not winning bars / total bars)")
        print("• Profit Factor = Sum of Winning Trade Returns / |Sum of Losing Trade Returns|")
        print("• Returns account for execution costs on entry and exit")


# ## Visualizations

# In[31]:


# Visualizations for single or multiple strategy types
if len(performance_df) > 0:
    if performance_df['strategy_type'].nunique() > 1:
        # Multiple strategy types - original visualization
        plt.figure(figsize=(14, 6))

        # Box plot of Sharpe by type
        plt.subplot(1, 2, 1)
        performance_df.boxplot(column='sharpe_ratio', by='strategy_type', ax=plt.gca())
        plt.xticks(rotation=45, ha='right')
        plt.title('Sharpe Ratio Distribution by Strategy Type')
        plt.suptitle('')  # Remove default title
        plt.ylabel('Sharpe Ratio')

        # Scatter: Return vs Sharpe
        plt.subplot(1, 2, 2)
        for stype in performance_df['strategy_type'].unique():
            mask = performance_df['strategy_type'] == stype
            plt.scatter(performance_df.loc[mask, 'total_return'], 
                       performance_df.loc[mask, 'sharpe_ratio'],
                       label=stype, alpha=0.6)
        plt.xlabel('Total Return')
        plt.ylabel('Sharpe Ratio')
        plt.title('Return vs Risk-Adjusted Return')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        # Single strategy type - parameter analysis visualization
        plt.figure(figsize=(15, 10))

        # 1. Sharpe ratio distribution
        plt.subplot(2, 2, 1)
        performance_df['sharpe_ratio'].hist(bins=20, alpha=0.7, color='blue')
        plt.axvline(performance_df['sharpe_ratio'].mean(), color='red', linestyle='--', label=f'Mean: {performance_df["sharpe_ratio"].mean():.2f}')
        plt.axvline(performance_df['sharpe_ratio'].median(), color='green', linestyle='--', label=f'Median: {performance_df["sharpe_ratio"].median():.2f}')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Count')
        plt.title('Sharpe Ratio Distribution')
        plt.legend()

        # 2. Return vs Sharpe scatter
        plt.subplot(2, 2, 2)
        # Determine which parameters exist (check both naming conventions)
        param_cols = [col for col in performance_df.columns if col.startswith('param_')]
        direct_param_cols = ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier', 'exit_threshold']
        available_param_cols = [col for col in direct_param_cols if col in performance_df.columns]

        # Use direct parameter names if available, otherwise fall back to param_ prefix
        if available_param_cols:
            param_cols = available_param_cols

        if len(param_cols) >= 2:
            # Use first two parameters for visualization
            scatter = plt.scatter(performance_df['total_return'], 
                                 performance_df['sharpe_ratio'],
                                 c=performance_df[param_cols[0]], 
                                 cmap='viridis',
                                 s=performance_df[param_cols[1]]*50 if performance_df[param_cols[1]].max() < 10 else 50,
                                 alpha=0.6)
            plt.colorbar(scatter, label=param_cols[0].replace('param_', ''))
            plt.title(f'Return vs Risk-Adjusted Return\n(Color={param_cols[0].replace("param_", "")}, Size={param_cols[1].replace("param_", "")})')
        else:
            plt.scatter(performance_df['total_return'], 
                       performance_df['sharpe_ratio'],
                       alpha=0.6)
            plt.title('Return vs Risk-Adjusted Return')
        plt.xlabel('Total Return')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True, alpha=0.3)

        # 3. Parameter heatmap (if enough data and two numeric parameters)
        if len(performance_df) > 10 and len(param_cols) >= 2:
            plt.subplot(2, 2, 3)
            try:
                # Create pivot table for heatmap
                pivot_sharpe = performance_df.pivot_table(
                    values='sharpe_ratio', 
                    index=param_cols[0], 
                    columns=param_cols[1],
                    aggfunc='mean'
                )
                if not pivot_sharpe.empty and pivot_sharpe.shape[0] > 1 and pivot_sharpe.shape[1] > 1:
                    sns.heatmap(pivot_sharpe, cmap='RdYlGn', center=0, 
                               cbar_kws={'label': 'Sharpe Ratio'})
                    plt.title(f'Sharpe Ratio by {param_cols[0].replace("param_", "")} and {param_cols[1].replace("param_", "")}')
            except:
                plt.text(0.5, 0.5, 'Not enough data for heatmap', 
                        ha='center', va='center', transform=plt.gca().transAxes)

        # 4. Box plot of returns
        plt.subplot(2, 2, 4)
        performance_df.boxplot(column=['total_return', 'sharpe_ratio'])
        plt.xticks(rotation=45)
        plt.title('Performance Metrics Distribution')
        plt.ylabel('Value')

        plt.tight_layout()
        plt.show()

        # Additional parameter analysis
        if param_cols:
            print("\n📈 Parameter Analysis:")
            for param in param_cols[:3]:  # Analyze first 3 parameters
                if param in performance_df.columns and performance_df[param].notna().any():
                    corr = performance_df[param].corr(performance_df['sharpe_ratio'])
                    param_display = param.replace('param_', '')
                    print(f"Correlation between {param_display} and Sharpe: {corr:.3f}")

            # Group by parameter ranges to find stable regions
            if len(param_cols) >= 2 and len(performance_df) > 20:
                print("\n🎯 Performance by Parameter Ranges:")
                try:
                    # Find numeric parameter columns
                    numeric_params = []
                    for col in param_cols:
                        if pd.api.types.is_numeric_dtype(performance_df[col]) and performance_df[col].notna().sum() > 0:
                            numeric_params.append(col)

                    if len(numeric_params) >= 2:
                        # Create bins for numeric parameters
                        param1_groups = pd.cut(performance_df[numeric_params[0]], bins=5)
                        param2_groups = pd.cut(performance_df[numeric_params[1]], bins=5)

                        param_summary = performance_df.groupby([param1_groups, param2_groups])['sharpe_ratio'].agg(['mean', 'std', 'count'])
                        param_summary = param_summary[param_summary['count'] > 0].sort_values('mean', ascending=False)

                        # Display with clean parameter names
                        param1_name = numeric_params[0].replace('param_', '')
                        param2_name = numeric_params[1].replace('param_', '')
                        print(f"\nTop performing {param1_name} x {param2_name} ranges:")
                        print(param_summary.head(10))
                    else:
                        print("Not enough numeric parameters for range analysis")
                except Exception as e:
                    print(f"Could not create parameter range analysis: {e}")


# ## Correlation Analysis for Ensemble Building

# In[32]:


def calculate_strategy_correlations(strategies_df, market_data, run_dir):
    """Calculate correlation matrix between strategies"""
    returns_dict = {}

    for idx, row in strategies_df.iterrows():
        try:
            # Use the global run_dir
            signals_path = run_dir / row['trace_path']
            signals = pd.read_parquet(signals_path)
            signals['ts'] = pd.to_datetime(signals['ts'])

            # Merge and calculate returns
            df = market_data.merge(signals[['ts', 'val']], left_on='timestamp', right_on='ts', how='left')
            df['signal'] = df['val'].ffill().fillna(0)
            df['returns'] = df['close'].pct_change()
            df['strategy_returns'] = df['returns'] * df['signal'].shift(1)

            returns_dict[row['strategy_hash']] = df['strategy_returns']
        except:
            pass

    # Create returns DataFrame and calculate correlation
    if returns_dict:
        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr()
    return pd.DataFrame()


# ## Enhanced Analysis: Stop Loss Impact & Trade Verification

# In[33]:


# Stop Loss Analysis for Top Strategies
if analyze_stop_losses and len(performance_df) > 0 and len(top_overall) > 0:
    print("\n📊 Stop Loss Impact Analysis")
    print("=" * 60)
    print("Using proper intraday stop loss simulation with high/low price data")

    stop_loss_results = {}

    # Analyze top 5 strategies
    for idx, (_, strategy) in enumerate(top_overall.head(5).iterrows()):
        print(f"\nAnalyzing stop losses for strategy {idx+1}: {strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")

        # Extract trades for this strategy
        trades = extract_trades(strategy['strategy_hash'], strategy['trace_path'], market_data, execution_cost_bps)

        if len(trades) > 0:
            # Calculate stop loss impact with proper intraday simulation
            sl_impact = calculate_stop_loss_impact(trades, stop_loss_levels, market_data)
            stop_loss_results[strategy['strategy_hash']] = sl_impact

            # Find optimal stop loss
            optimal_sl = sl_impact.loc[sl_impact['total_return'].idxmax()]
            current_return = trades['net_return'].sum()

            print(f"  Current total return: {current_return*100:.2f}%")
            print(f"  Optimal stop loss: {optimal_sl['stop_loss_pct']:.2f}% → Return: {optimal_sl['total_return']*100:.2f}%")
            print(f"  Improvement: {(optimal_sl['total_return'] - current_return)*100:.2f}%")
            print(f"  Trades stopped out: {optimal_sl['stopped_out_count']} ({optimal_sl['stopped_out_rate']*100:.1f}%)")

            # Show impact on winners vs losers
            if 'avg_winner' in optimal_sl and 'avg_loser' in optimal_sl:
                print(f"  Average winner: {optimal_sl['avg_winner']*100:.2f}%")
                print(f"  Average loser: {optimal_sl['avg_loser']*100:.2f}%")

    # Visualize stop loss impact
    if stop_loss_results:
        plt.figure(figsize=(15, 10))

        # Plot 1: Total return vs stop loss
        plt.subplot(2, 2, 1)
        for i, (hash_id, sl_df) in enumerate(stop_loss_results.items()):
            plt.plot(sl_df['stop_loss_pct'], sl_df['total_return'] * 100, 
                    label=f'Strategy {i+1}', marker='o', markersize=4)

        plt.xlabel('Stop Loss (%)')
        plt.ylabel('Total Return (%)')
        plt.title('Stop Loss Impact on Total Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Win rate impact
        plt.subplot(2, 2, 2)
        for i, (hash_id, sl_df) in enumerate(stop_loss_results.items()):
            plt.plot(sl_df['stop_loss_pct'], sl_df['win_rate'] * 100, 
                    label=f'Strategy {i+1}', marker='o', markersize=4)

        plt.xlabel('Stop Loss (%)')
        plt.ylabel('Win Rate (%)')
        plt.title('Stop Loss Impact on Win Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Stopped out rate
        plt.subplot(2, 2, 3)
        for i, (hash_id, sl_df) in enumerate(stop_loss_results.items()):
            plt.plot(sl_df['stop_loss_pct'], sl_df['stopped_out_rate'] * 100, 
                    label=f'Strategy {i+1}', marker='o', markersize=4)

        plt.xlabel('Stop Loss (%)')
        plt.ylabel('Stopped Out Rate (%)')
        plt.title('Percentage of Trades Hitting Stop Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Average return per trade
        plt.subplot(2, 2, 4)
        for i, (hash_id, sl_df) in enumerate(stop_loss_results.items()):
            plt.plot(sl_df['stop_loss_pct'], sl_df['avg_return_per_trade'] * 100, 
                    label=f'Strategy {i+1}', marker='o', markersize=4)

        plt.xlabel('Stop Loss (%)')
        plt.ylabel('Avg Return per Trade (%)')
        plt.title('Average Return per Trade with Stop Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Summary recommendations
        print("\n🎯 Stop Loss Recommendations:")
        print("Based on the analysis across top strategies:")

        # Find the stop loss level with best average performance
        avg_returns_by_sl = {}
        for sl_level in stop_loss_levels:
            returns_at_level = []
            for hash_id, sl_df in stop_loss_results.items():
                sl_row = sl_df[sl_df['stop_loss_pct'] == sl_level]
                if len(sl_row) > 0:
                    returns_at_level.append(sl_row.iloc[0]['total_return'])
            if returns_at_level:
                avg_returns_by_sl[sl_level] = np.mean(returns_at_level)

        if avg_returns_by_sl:
            best_sl = max(avg_returns_by_sl, key=avg_returns_by_sl.get)
            print(f"\n• Optimal stop loss across strategies: {best_sl:.2f}%")

        print("\nKey insights:")
        print("• Very tight stop losses (0.01-0.05%) will trigger on normal market noise")
        print("• The apparent improvement from tight stops in the flawed analysis was misleading")
        print("• Proper stop losses should balance downside protection with allowing winners to run")
        print("• Consider volatility-adjusted stops rather than fixed percentage stops")
else:
    print("\n⚠️ Skipping stop loss analysis")


# In[34]:


# Intraday Constraint Verification
if verify_intraday and len(performance_df) > 0 and len(top_overall) > 0:
    print("\n⏰ Intraday Constraint Verification")
    print("=" * 60)
    print(f"Market timezone: {market_timezone}")

    constraint_violations = []

    # Check top 10 strategies
    for idx, (_, strategy) in enumerate(top_overall.head(10).iterrows()):
        # Extract trades
        trades = extract_trades(strategy['strategy_hash'], strategy['trace_path'], market_data, execution_cost_bps)

        if len(trades) > 0:
            # Verify constraints
            constraints = verify_intraday_constraint(trades, market_timezone)

            if constraints['overnight_positions'] > 0 or constraints['after_hours_entries'] > 0 or constraints['after_hours_exits'] > 0:
                constraint_violations.append({
                    'strategy': f"{strategy['strategy_type']} - {strategy['strategy_hash'][:8]}",
                    'overnight': constraints['overnight_positions'],
                    'overnight_pct': constraints['overnight_position_pct'],
                    'after_hours_entries': constraints['after_hours_entries'],
                    'after_hours_exits': constraints['after_hours_exits']
                })

                print(f"\n⚠️ Strategy {idx+1} has constraint violations:")
                print(f"   Overnight positions: {constraints['overnight_positions']} ({constraints['overnight_position_pct']:.1f}%)")
                print(f"   After-hours entries: {constraints['after_hours_entries']}")
                print(f"   After-hours exits: {constraints['after_hours_exits']}")
            else:
                print(f"\n✅ Strategy {idx+1}: All trades respect intraday constraints")

            # Show trade duration statistics
            print(f"   Avg duration: {constraints['avg_trade_duration_minutes']:.1f} minutes")
            print(f"   Max duration: {constraints['max_trade_duration_minutes']:.1f} minutes")
            print(f"   Trades > 390 min: {constraints['trades_over_390_minutes']}")

    # Summary
    if constraint_violations:
        print(f"\n⚠️ Found {len(constraint_violations)} strategies with constraint violations")
        violations_df = pd.DataFrame(constraint_violations)
        print("\nViolation Summary:")
        print(violations_df.to_string(index=False))
    else:
        print("\n✅ All top 10 strategies respect intraday constraints!")

    # Visualize entry/exit times for a sample strategy
    if len(top_overall) > 0:
        sample_strategy = top_overall.iloc[0]
        sample_trades = extract_trades(sample_strategy['strategy_hash'], sample_strategy['trace_path'], market_data, execution_cost_bps)

        if len(sample_trades) > 0:
            sample_constraints = verify_intraday_constraint(sample_trades, market_timezone)

            if 'entries_by_hour' in sample_constraints and sample_constraints['entries_by_hour']:
                plt.figure(figsize=(12, 5))

                # Entry times
                plt.subplot(1, 2, 1)
                hours = sorted(sample_constraints['entries_by_hour'].keys())
                counts = [sample_constraints['entries_by_hour'][h] for h in hours]
                plt.bar(hours, counts)
                plt.axvline(9.5, color='red', linestyle='--', alpha=0.5, label='Market Open')
                plt.axvline(16, color='red', linestyle='--', alpha=0.5, label='Market Close')
                plt.xlabel('Hour of Day')
                plt.ylabel('Number of Entries')
                plt.title('Trade Entry Times (Top Strategy)')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Exit times
                plt.subplot(1, 2, 2)
                if 'exits_by_hour' in sample_constraints and sample_constraints['exits_by_hour']:
                    hours = sorted(sample_constraints['exits_by_hour'].keys())
                    counts = [sample_constraints['exits_by_hour'][h] for h in hours]
                    plt.bar(hours, counts)
                    plt.axvline(9.5, color='red', linestyle='--', alpha=0.5, label='Market Open')
                    plt.axvline(16, color='red', linestyle='--', alpha=0.5, label='Market Close')
                    plt.xlabel('Hour of Day')
                    plt.ylabel('Number of Exits')
                    plt.title('Trade Exit Times (Top Strategy)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()
else:
    print("\n⚠️ Skipping intraday constraint verification")


# In[35]:


# Optimized correlation calculation with progress tracking
if len(performance_df) > 0 and len(top_overall) > 1:
    print("\n🔗 Calculating correlations among top strategies...")
    print(f"Processing {len(top_overall)} strategies...")

    # First, load all returns data in one pass
    returns_dict = {}

    for idx, (_, row) in enumerate(top_overall.iterrows()):
        if idx % 5 == 0:
            print(f"  Loading signals: {idx}/{len(top_overall)}")

        try:
            signals_path = run_dir / row['trace_path']

            # Load signals
            signals = pd.read_parquet(signals_path)
            signals['ts'] = pd.to_datetime(signals['ts'])

            # Merge with market data (already in memory)
            df = market_data.merge(
                signals[['ts', 'val']], 
                left_on='timestamp', 
                right_on='ts', 
                how='left'
            )
            df['signal'] = df['val'].ffill().fillna(0)

            # Calculate strategy returns only once
            df['returns'] = df['close'].pct_change()
            df['strategy_returns'] = df['returns'] * df['signal'].shift(1)

            returns_dict[row['strategy_hash']] = df['strategy_returns'].values
        except Exception as e:
            print(f"  Warning: Could not load {row['strategy_hash'][:8]}: {e}")

    print(f"✅ Loaded returns for {len(returns_dict)} strategies")

    if len(returns_dict) >= 2:
        # Convert to DataFrame for correlation calculation
        returns_df = pd.DataFrame(returns_dict)

        # Calculate correlation matrix (this is fast once data is loaded)
        print("Calculating correlation matrix...")
        corr_matrix = returns_df.corr()

        # Find uncorrelated pairs
        uncorrelated_pairs = []
        n = len(corr_matrix)
        total_pairs = n * (n - 1) // 2

        pair_count = 0
        for i in range(n):
            for j in range(i+1, n):
                pair_count += 1

                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) < correlation_threshold:
                    uncorrelated_pairs.append({
                        'strategy1': corr_matrix.index[i],
                        'strategy2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        print(f"✅ Found {len(uncorrelated_pairs)} uncorrelated pairs (correlation < {correlation_threshold})")

        # Visualize correlation matrix
        if len(corr_matrix) <= 20:
            plt.figure(figsize=(10, 8))
            # Only show annotations if matrix is small enough
            show_annot = len(corr_matrix) <= 10
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
                       xticklabels=[h[:8] for h in corr_matrix.columns],
                       yticklabels=[h[:8] for h in corr_matrix.index],
                       annot=show_annot, fmt='.2f' if show_annot else None)
            plt.title('Strategy Correlation Matrix')
            plt.tight_layout()
            plt.show()

            # Show correlation statistics
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            print(f"\nCorrelation Statistics:")
            print(f"  Mean correlation: {np.mean(corr_values):.3f}")
            print(f"  Median correlation: {np.median(corr_values):.3f}")
            print(f"  Min correlation: {np.min(corr_values):.3f}")
            print(f"  Max correlation: {np.max(corr_values):.3f}")
        else:
            print(f"Skipping heatmap visualization (too many strategies: {len(corr_matrix)})")
    else:
        print("❌ Not enough strategies loaded for correlation analysis")


# ## Ensemble Recommendations

# In[36]:


# Build optimal ensemble
if len(performance_df) > 0 and 'corr_matrix' in locals() and not corr_matrix.empty:
    # Start with best strategy
    ensemble = [top_overall.iloc[0]['strategy_hash']]
    ensemble_data = [top_overall.iloc[0]]

    # Add uncorrelated strategies
    for idx, candidate in top_overall.iloc[1:].iterrows():
        if len(ensemble) >= ensemble_size:
            break

        # Check correlation with existing ensemble members
        candidate_hash = candidate['strategy_hash']
        if candidate_hash in corr_matrix.columns:
            max_corr = 0
            for existing in ensemble:
                if existing in corr_matrix.index:
                    corr = abs(corr_matrix.loc[existing, candidate_hash])
                    max_corr = max(max_corr, corr)

            if max_corr < correlation_threshold:
                ensemble.append(candidate_hash)
                ensemble_data.append(candidate)

    print(f"\n🎯 Recommended Ensemble ({len(ensemble)} strategies):")
    print("=" * 80)

    ensemble_df = pd.DataFrame(ensemble_data)
    for idx, row in ensemble_df.iterrows():
        print(f"\n{idx+1}. {row['strategy_type']} - {row['strategy_hash'][:8]}")
        print(f"   Sharpe: {row['sharpe_ratio']:.2f} | Return: {row['total_return']:.1%}")

    # Calculate ensemble metrics
    print(f"\nEnsemble Statistics:")
    print(f"  Average Sharpe: {ensemble_df['sharpe_ratio'].mean():.2f}")
    print(f"  Average Return: {ensemble_df['total_return'].mean():.1%}")
    print(f"  Strategy Types: {', '.join(ensemble_df['strategy_type'].unique())}")


# In[37]:


# Export recommendations with enhanced metrics
if len(performance_df) > 0:
    # Helper function to convert numpy types to Python native types
    def convert_to_native(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        else:
            return obj

    recommendations = {
        'run_info': {
            'run_id': run_dir.name,
            'config_name': config_name,
            'generated_at': datetime.now().isoformat(),
            'total_strategies': len(strategy_index) if strategy_index is not None else 0,
            'strategies_analyzed': len(performance_df),
            'execution_cost_bps': execution_cost_bps
        },
        'best_individual': {},
        'best_by_type': {},
        'ensemble': [],
        'stop_loss_recommendations': {
            'conservative': 0.25,
            'balanced': 0.50,
            'aggressive': 1.00
        }
    }

    # Best overall
    if len(top_overall) > 0:
        best = top_overall.iloc[0]
        recommendations['best_individual'] = {
            'strategy_hash': best['strategy_hash'],
            'strategy_type': best['strategy_type'],
            'sharpe_ratio': float(best['sharpe_ratio']),
            'total_return': float(best['total_return']),
            'max_drawdown': float(best['max_drawdown']),
            'win_rate': float(best.get('win_rate', 0)),
            'avg_return_per_trade': float(best.get('avg_return_per_trade', 0)),
            'num_trades': int(best.get('num_trades', 0)),
            'total_execution_cost': float(best.get('total_execution_cost', 0)),
            'parameters': {col.replace('param_', ''): convert_to_native(best[col]) 
                           for col in best.index if (col.startswith('param_') or col in ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier'])
                           and pd.notna(best[col])}
        }

    # Best by type
    for stype in performance_df['strategy_type'].unique():
        type_best = performance_df[performance_df['strategy_type'] == stype].nlargest(1, 'sharpe_ratio')
        if len(type_best) > 0:
            row = type_best.iloc[0]
            recommendations['best_by_type'][stype] = {
                'strategy_hash': row['strategy_hash'],
                'sharpe_ratio': float(row['sharpe_ratio']),
                'total_return': float(row['total_return']),
                'win_rate': float(row.get('win_rate', 0)),
                'avg_return_per_trade': float(row.get('avg_return_per_trade', 0))
            }

    # Ensemble
    if 'ensemble_df' in locals():
        for idx, row in ensemble_df.iterrows():
            recommendations['ensemble'].append({
                'strategy_hash': row['strategy_hash'],
                'strategy_type': row['strategy_type'],
                'sharpe_ratio': float(row['sharpe_ratio']),
                'win_rate': float(row.get('win_rate', 0)),
                'weight': 1.0 / len(ensemble_df)  # Equal weight for now
            })

    # Add stop loss analysis results if available
    if 'stop_loss_results' in locals() and stop_loss_results:
        recommendations['stop_loss_analysis'] = {}
        for hash_id, sl_df in list(stop_loss_results.items())[:3]:  # Top 3 strategies
            optimal_idx = sl_df['total_return'].idxmax()
            recommendations['stop_loss_analysis'][hash_id[:8]] = {
                'optimal_stop_loss_pct': float(sl_df.loc[optimal_idx, 'stop_loss_pct']),
                'optimal_total_return': float(sl_df.loc[optimal_idx, 'total_return']),
                'improvement_pct': float((sl_df.loc[optimal_idx, 'total_return'] - sl_df.loc[0, 'total_return']) * 100)
            }

    # Convert all to native Python types before saving
    recommendations = convert_to_native(recommendations)

    # Save files
    with open(run_dir / 'recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)

    performance_df.to_csv(run_dir / 'performance_analysis.csv', index=False)

    # Also save enhanced metrics
    enhanced_metrics_df = performance_df[['strategy_hash', 'strategy_type', 'sharpe_ratio', 'total_return',
                                          'win_rate', 'avg_return_per_trade', 'num_trades', 'total_execution_cost']].copy()
    enhanced_metrics_df.to_csv(run_dir / 'enhanced_metrics.csv', index=False)

    print("\n✅ Results exported:")
    print(f"  - recommendations.json (with enhanced metrics)")
    print(f"  - performance_analysis.csv")
    print(f"  - enhanced_metrics.csv")
else:
    print("⚠️ No results to export")


# ## Further Analysis Options
# 
# You can extend this analysis using the available snippets and queries:
# 
# ### Regime Analysis
# To analyze strategy performance under different market regimes or with filters:
# 
# ```python
# # Load and run regime analysis
# %load /Users/daws/ADMF-PC/src/analytics/snippets/regime/volatility_regimes.py
# ```
# 
# ### Filter Effectiveness
# To evaluate how different filters affect strategy performance:
# 
# ```python
# # Analyze filter impact
# %load /Users/daws/ADMF-PC/src/analytics/snippets/filters/filter_effectiveness.py
# ```
# 
# ### Custom SQL Queries
# For direct database analysis of traces:
# 
# ```python
# # Run custom DuckDB queries
# %load /Users/daws/ADMF-PC/src/analytics/queries/signal_patterns.sql
# ```
# 
# ### Available Analysis Snippets:
# - **Exploratory**: signal_frequency.py, trade_duration.py, position_distribution.py
# - **Ensembles**: find_uncorrelated.py, optimal_weights.py, ensemble_backtest.py
# - **Regime**: volatility_regimes.py, trend_regimes.py, time_of_day.py
# - **Filters**: filter_effectiveness.py, filter_combinations.py
# - **Risk**: drawdown_analysis.py, risk_metrics.py, position_sizing.py
# 
# Use `%run` instead of `%load` to execute immediately, or modify the loaded code as needed.

# ## Summary
# 
# Analysis complete! Key files generated:
# - `recommendations.json` - Best strategies and ensemble recommendations
# - `performance_analysis.csv` - Full performance data for all strategies
# 
# Next steps:
# 1. Use the recommended ensemble for live trading
# 2. Deep dive into specific strategy types if needed
# 3. Run regime-specific analysis to understand performance drivers

# In[39]:


# %load /Users/daws/ADMF-PC/src/analytics/snippets/micro_movement_analysis.py
# Micro-Movement Trading Analysis
# Optimized for strategies with <0.2% average movements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration for micro-movements
MICRO_STOP_LEVELS = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]
PROFIT_TARGET_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def analyze_micro_movements(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Analyze strategies with very small price movements
    Tests both tight stops and profit targets
    """
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)

    if len(trades) == 0:
        return None

    results = []

    # Test stop loss + profit target combinations
    for stop_pct in MICRO_STOP_LEVELS:
        for target_pct in PROFIT_TARGET_LEVELS:
            if target_pct <= stop_pct:
                continue  # Skip invalid combinations

            trades_modified = []
            stops_hit = 0
            targets_hit = 0

            for _, trade in trades.iterrows():
                # Get intraday prices
                trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]

                if len(trade_prices) == 0:
                    continue

                entry_price = trade['entry_price']
                direction = trade['direction']

                # Calculate stop and target prices
                if direction == 1:  # Long
                    stop_price = entry_price * (1 - stop_pct/100)
                    target_price = entry_price * (1 + target_pct/100)
                else:  # Short
                    stop_price = entry_price * (1 + stop_pct/100)
                    target_price = entry_price * (1 - target_pct/100)

                # Check each bar for stop or target hit
                exit_price = trade['exit_price']
                exit_type = 'signal'

                for idx, bar in trade_prices.iterrows():
                    if direction == 1:  # Long
                        if bar['low'] <= stop_price:
                            exit_price = stop_price
                            exit_type = 'stop'
                            stops_hit += 1
                            break
                        elif bar['high'] >= target_price:
                            exit_price = target_price
                            exit_type = 'target'
                            targets_hit += 1
                            break
                    else:  # Short
                        if bar['high'] >= stop_price:
                            exit_price = stop_price
                            exit_type = 'stop'
                            stops_hit += 1
                            break
                        elif bar['low'] <= target_price:
                            exit_price = target_price
                            exit_type = 'target'
                            targets_hit += 1
                            break

                # Calculate return
                if direction == 1:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price

                net_return = raw_return - trade['execution_cost']

                trades_modified.append({
                    'net_return': net_return,
                    'exit_type': exit_type
                })

            trades_df = pd.DataFrame(trades_modified)

            if len(trades_df) > 0:
                # Calculate metrics
                total_return = (1 + trades_df['net_return']).cumprod().iloc[-1] - 1
                win_rate = (trades_df['net_return'] > 0).mean()

                # Calculate Sharpe
                if trades_df['net_return'].std() > 0:
                    sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * 78)
                else:
                    sharpe = 0

                results.append({
                    'stop_pct': stop_pct,
                    'target_pct': target_pct,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'win_rate': win_rate,
                    'stops_hit': stops_hit,
                    'targets_hit': targets_hit,
                    'stops_pct': stops_hit / len(trades_df) * 100,
                    'targets_pct': targets_hit / len(trades_df) * 100,
                    'signal_exits_pct': (len(trades_df) - stops_hit - targets_hit) / len(trades_df) * 100
                })

    return pd.DataFrame(results)

# Main analysis
if len(performance_df) > 0:
    print("🔬 Micro-Movement Trading Analysis")
    print("=" * 80)
    print(f"Testing stop levels: {MICRO_STOP_LEVELS}")
    print(f"Testing profit targets: {PROFIT_TARGET_LEVELS}")

    # Get high-frequency strategies
    trading_days = len(market_data['timestamp'].dt.date.unique())
    high_freq_df = performance_df[performance_df['num_trades'] >= 2 * trading_days]

    if len(high_freq_df) > 0:
        # Analyze top 5 high-frequency strategies
        print("\n📊 Analyzing Stop + Target Combinations:")

        all_results = []

        for idx, row in high_freq_df.head(5).iterrows():
            print(f"\nStrategy {idx+1}: {row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"  Base performance: Sharpe={row['sharpe_ratio']:.2f}, Return={row['total_return']*100:.2f}%")

            # Analyze with micro stops and targets
            micro_results = analyze_micro_movements(
                row['strategy_hash'],
                row['trace_path'],
                market_data,
                execution_cost_bps
            )

            if micro_results is not None and len(micro_results) > 0:
                # Find optimal combination
                optimal_idx = micro_results['sharpe_ratio'].idxmax()
                optimal = micro_results.iloc[optimal_idx]

                print(f"  Optimal: Stop={optimal['stop_pct']:.3f}%, Target={optimal['target_pct']:.2f}%")
                print(f"  New Sharpe: {optimal['sharpe_ratio']:.2f} (was {row['sharpe_ratio']:.2f})")
                print(f"  New Return: {optimal['total_return']*100:.2f}% (was {row['total_return']*100:.2f}%)")
                print(f"  Exit breakdown: Stops={optimal['stops_pct']:.1f}%, Targets={optimal['targets_pct']:.1f}%, Signal={optimal['signal_exits_pct']:.1f}%")

                micro_results['strategy_hash'] = row['strategy_hash'][:8]
                all_results.append(micro_results)

        # Visualize results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)

            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. Sharpe ratio heatmap (for first strategy)
            ax = axes[0, 0]
            first_strategy = all_results[0]
            pivot_sharpe = first_strategy.pivot_table(
                values='sharpe_ratio',
                index='stop_pct',
                columns='target_pct'
            )
            sns.heatmap(pivot_sharpe, cmap='RdYlGn', center=0, ax=ax, 
                       cbar_kws={'label': 'Sharpe Ratio'})
            ax.set_title('Sharpe Ratio by Stop/Target Combination')
            ax.set_xlabel('Profit Target %')
            ax.set_ylabel('Stop Loss %')

            # 2. Win rate heatmap
            ax = axes[0, 1]
            pivot_winrate = first_strategy.pivot_table(
                values='win_rate',
                index='stop_pct',
                columns='target_pct'
            )
            sns.heatmap(pivot_winrate * 100, cmap='RdYlGn', center=50, ax=ax,
                       cbar_kws={'label': 'Win Rate %'})
            ax.set_title('Win Rate by Stop/Target Combination')
            ax.set_xlabel('Profit Target %')
            ax.set_ylabel('Stop Loss %')

            # 3. Exit type distribution
            ax = axes[1, 0]
            # Average across all strategies
            avg_exits = combined_results.groupby(['stop_pct', 'target_pct'])[['stops_pct', 'targets_pct', 'signal_exits_pct']].mean()
            optimal_combos = combined_results.groupby('strategy_hash')['sharpe_ratio'].idxmax()

            # Show exit distribution for optimal combinations
            exit_data = []
            for strategy, idx in optimal_combos.items():
                row = combined_results.iloc[idx]
                exit_data.append({
                    'Strategy': strategy,
                    'Stops': row['stops_pct'],
                    'Targets': row['targets_pct'],
                    'Signal': row['signal_exits_pct']
                })

            exit_df = pd.DataFrame(exit_data)
            exit_df.set_index('Strategy').plot(kind='bar', stacked=True, ax=ax)
            ax.set_ylabel('Exit Type %')
            ax.set_title('Exit Type Distribution (Optimal Settings)')
            ax.legend(title='Exit Type')

            # 4. Improvement summary
            ax = axes[1, 1]
            ax.axis('off')

            summary_text = "Optimal Stop/Target Combinations:\n\n"
            for strategy in combined_results['strategy_hash'].unique()[:5]:
                strategy_data = combined_results[combined_results['strategy_hash'] == strategy]
                if len(strategy_data) > 0:
                    optimal_idx = strategy_data['sharpe_ratio'].idxmax()
                    optimal = strategy_data.iloc[optimal_idx]

                    # Get original performance
                    orig = high_freq_df[high_freq_df['strategy_hash'].str.contains(strategy)]
                if len(orig) > 0:
                    orig_sharpe = orig.iloc[0]['sharpe_ratio']
                    improvement = optimal['sharpe_ratio'] - orig_sharpe

                    summary_text += f"{strategy}:\n"
                    summary_text += f"  Stop: {optimal['stop_pct']:.3f}%, Target: {optimal['target_pct']:.2f}%\n"
                    summary_text += f"  Sharpe: {orig_sharpe:.2f} → {optimal['sharpe_ratio']:.2f} "
                    summary_text += f"({improvement:+.2f})\n\n"

            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=11, family='monospace')

            plt.tight_layout()
            plt.show()

            # Key insights
            print("\n💡 Key Insights:")
            print("=" * 60)

            # Find most common optimal settings
            optimal_stops = []
            optimal_targets = []
            for strategy in combined_results['strategy_hash'].unique():
                strategy_data = combined_results[combined_results['strategy_hash'] == strategy]
                optimal_idx = strategy_data['sharpe_ratio'].idxmax()
                optimal = strategy_data.iloc[optimal_idx]
                optimal_stops.append(optimal['stop_pct'])
                optimal_targets.append(optimal['target_pct'])

            print(f"1. Most common optimal stop: {np.median(optimal_stops):.3f}%")
            print(f"2. Most common optimal target: {np.median(optimal_targets):.2f}%")
            print(f"3. Stop/Target ratio: ~1:{np.median(optimal_targets)/np.median(optimal_stops):.1f}")

            # Save results
            combined_results.to_csv(run_dir / 'micro_movement_analysis.csv', index=False)
            print(f"\n✅ Saved analysis to: micro_movement_analysis.csv")

            print("\n🎯 Recommendations:")
            print("1. Use very tight stops (0.1-0.2%) to limit losses")
            print("2. Set profit targets at 2-3x stop distance")
            print("3. Focus on high win rate rather than big moves")
            print("4. Consider commission impact on such small moves")

    else:
        print("❌ No high-frequency strategies found")


# In[41]:


# %load /Users/daws/ADMF-PC/src/analytics/snippets/complete_stop_target_analysis.py
# Complete Stop/Target Analysis - All Inclusive
# Tests stop loss + profit target combinations on all strategies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
STOP_TARGET_PAIRS = [
    (0.05, 0.10),    # 2:1 reward/risk
    (0.075, 0.10),   # 1.33:1 (optimal from training)
    (0.10, 0.15),    # 1.5:1
    (0.10, 0.20),    # 2:1
    (0.15, 0.30),    # 2:1
    (0.20, 0.40),    # 2:1
    (0, 0),          # No stop/target (baseline)
]

def apply_stop_target(trades_df, stop_pct, target_pct, market_data):
    """Apply stop loss and profit target to trades"""
    if stop_pct == 0 and target_pct == 0:
        # No modification - return original
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

        # Set stop and target prices
        if target_pct > 0:  # Use profit target
            if direction == 1:  # Long
                stop_price = entry_price * (1 - stop_pct/100) if stop_pct > 0 else 0
                target_price = entry_price * (1 + target_pct/100)
            else:  # Short
                stop_price = entry_price * (1 + stop_pct/100) if stop_pct > 0 else float('inf')
                target_price = entry_price * (1 - target_pct/100)
        else:  # Stop only
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

        # Calculate return
        if direction == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price

        net_return = raw_return - trade['execution_cost']
        modified_returns.append(net_return)

    return np.array(modified_returns), exit_types

# Main analysis
print("🎯 Complete Stop/Target Analysis")
print("=" * 80)

if len(performance_df) > 0:
    # Basic statistics
    trading_days = len(market_data['timestamp'].dt.date.unique())
    print(f"Dataset: {trading_days} trading days")
    print(f"Total strategies: {len(performance_df)}")

    # Trade frequency stats
    performance_df['trades_per_day'] = performance_df['num_trades'] / trading_days
    print(f"\nTrade frequency:")
    print(f"  Mean: {performance_df['trades_per_day'].mean():.2f} trades/day")
    print(f"  Max: {performance_df['trades_per_day'].max():.2f} trades/day")

    # Get top strategies (adjust number as needed)
    TOP_N = min(20, len(performance_df))
    top_strategies = performance_df.nlargest(TOP_N, 'num_trades')  # Sort by trade count for better statistics

    print(f"\nAnalyzing top {len(top_strategies)} strategies by trade count")

    # Analyze each strategy with different stop/target combinations
    all_results = []

    for idx, row in top_strategies.iterrows():
        # Extract trades once
        trades = extract_trades(row['strategy_hash'], row['trace_path'], market_data, execution_cost_bps)

        if len(trades) < 10:  # Skip if too few trades
            continue

        strategy_results = {
            'strategy_hash': row['strategy_hash'],
            'strategy_type': row['strategy_type'],
            'num_trades': len(trades),
            'trades_per_day': len(trades) / trading_days,
            'base_sharpe': row['sharpe_ratio'],
            'base_return': row['total_return'],
            'base_win_rate': row.get('win_rate', 0),
            'period': row.get('period', 'N/A'),
            'std_dev': row.get('std_dev', 'N/A')
        }

        # Test each stop/target combination
        for stop_pct, target_pct in STOP_TARGET_PAIRS:
            # Apply stop/target
            returns_array, exit_types = apply_stop_target(trades, stop_pct, target_pct, market_data)

            # Calculate metrics
            total_return = (1 + returns_array).prod() - 1
            win_rate = (returns_array > 0).mean()

            if returns_array.std() > 0:
                sharpe = returns_array.mean() / returns_array.std() * np.sqrt(252 * len(trades) / trading_days)
            else:
                sharpe = 0

            # Store results
            key = f"stop_{stop_pct}_target_{target_pct}"
            strategy_results[f"{key}_return"] = total_return
            strategy_results[f"{key}_sharpe"] = sharpe
            strategy_results[f"{key}_win_rate"] = win_rate
            strategy_results[f"{key}_stop_pct"] = exit_types['stop'] / len(returns_array) * 100
            strategy_results[f"{key}_target_pct"] = exit_types['target'] / len(returns_array) * 100

        all_results.append(strategy_results)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        # Find best stop/target for each strategy
        print("\n📊 Optimal Stop/Target Combinations:")
        print("=" * 80)

        optimal_configs = []

        for idx, row in results_df.iterrows():
            best_sharpe = row['base_sharpe']
            best_config = 'No stop/target'
            best_stop = 0
            best_target = 0

            for stop, target in STOP_TARGET_PAIRS:
                key = f"stop_{stop}_target_{target}"
                if f"{key}_sharpe" in row and row[f"{key}_sharpe"] > best_sharpe:
                    best_sharpe = row[f"{key}_sharpe"]
                    best_config = f"Stop={stop}%, Target={target}%"
                    best_stop = stop
                    best_target = target

            optimal_configs.append({
                'strategy': f"{row['strategy_type']}_{row['strategy_hash'][:8]}",
                'period': row['period'],
                'std_dev': row['std_dev'],
                'trades': row['num_trades'],
                'trades_per_day': row['trades_per_day'],
                'base_sharpe': row['base_sharpe'],
                'base_return': row['base_return'] * 100,
                'best_config': best_config,
                'best_sharpe': best_sharpe,
                'best_return': row[f"stop_{best_stop}_target_{best_target}_return"] * 100 if best_stop > 0 or best_target > 0 else row['base_return'] * 100,
                'improvement': best_sharpe - row['base_sharpe']
            })

        optimal_df = pd.DataFrame(optimal_configs)

        # Show top 10 improvements
        print("\nTop 10 strategies by Sharpe improvement:")
        top_improvements = optimal_df.nlargest(10, 'improvement')

        for idx, row in top_improvements.iterrows():
            print(f"\n{row['strategy']} (period={row['period']}, std_dev={row['std_dev']})")
            print(f"  Trades: {row['trades']} ({row['trades_per_day']:.1f}/day)")
            print(f"  Base: Sharpe={row['base_sharpe']:.2f}, Return={row['base_return']:.2f}%")
            print(f"  Best: {row['best_config']} → Sharpe={row['best_sharpe']:.2f}, Return={row['best_return']:.2f}%")
            print(f"  Improvement: Sharpe +{row['improvement']:.2f}")

        # Aggregate analysis
        print("\n📈 Aggregate Analysis:")
        print("=" * 60)

        # Which stop/target works best overall?
        config_performance = {}

        for stop, target in STOP_TARGET_PAIRS:
            key = f"stop_{stop}_target_{target}"
            sharpe_col = f"{key}_sharpe"

            if sharpe_col in results_df.columns:
                avg_sharpe = results_df[sharpe_col].mean()
                avg_return = results_df[f"{key}_return"].mean()
                win_count = (results_df[sharpe_col] > results_df['base_sharpe']).sum()

                config_performance[f"{stop}/{target}"] = {
                    'avg_sharpe': avg_sharpe,
                    'avg_return': avg_return * 100,
                    'win_count': win_count,
                    'win_rate': win_count / len(results_df) * 100
                }

        print("\nAverage performance by stop/target configuration:")
        config_df = pd.DataFrame(config_performance).T
        config_df = config_df.sort_values('avg_sharpe', ascending=False)

        for config, metrics in config_df.iterrows():
            print(f"\nStop/Target = {config}%:")
            print(f"  Avg Sharpe: {metrics['avg_sharpe']:.2f}")
            print(f"  Avg Return: {metrics['avg_return']:.2f}%")
            print(f"  Improves {metrics['win_count']:.0f}/{len(results_df)} strategies ({metrics['win_rate']:.1f}%)")

        # Test the specific 0.075/0.1 combination
        print("\n🎯 Focus: 0.075% Stop / 0.1% Target Performance:")
        print("=" * 60)

        key_075_10 = "stop_0.075_target_0.1"
        if f"{key_075_10}_sharpe" in results_df.columns:
            # Performance stats
            avg_return_075_10 = results_df[f"{key_075_10}_return"].mean() * 100
            avg_sharpe_075_10 = results_df[f"{key_075_10}_sharpe"].mean()
            avg_stop_rate = results_df[f"{key_075_10}_stop_pct"].mean()
            avg_target_rate = results_df[f"{key_075_10}_target_pct"].mean()

            print(f"Average return: {avg_return_075_10:.2f}%")
            print(f"Average Sharpe: {avg_sharpe_075_10:.2f}")
            print(f"Average stop hit rate: {avg_stop_rate:.1f}%")
            print(f"Average target hit rate: {avg_target_rate:.1f}%")

            # Compare to base
            avg_base_return = results_df['base_return'].mean() * 100
            avg_base_sharpe = results_df['base_sharpe'].mean()

            print(f"\nImprovement over base:")
            print(f"  Return: {avg_base_return:.2f}% → {avg_return_075_10:.2f}% ({avg_return_075_10 - avg_base_return:+.2f}%)")
            print(f"  Sharpe: {avg_base_sharpe:.2f} → {avg_sharpe_075_10:.2f} ({avg_sharpe_075_10 - avg_base_sharpe:+.2f})")

            # Best performers with this config
            print(f"\nTop 5 performers with 0.075/0.1 stop/target:")
            top_with_config = results_df.nlargest(5, f"{key_075_10}_sharpe")

            for idx, row in top_with_config.iterrows():
                print(f"\n{row['strategy_type']} (period={row['period']}, std_dev={row['std_dev']})")
                print(f"  Return: {row[f'{key_075_10}_return']*100:.2f}%")
                print(f"  Sharpe: {row[f'{key_075_10}_sharpe']:.2f}")
                print(f"  Win Rate: {row[f'{key_075_10}_win_rate']*100:.1f}%")
                print(f"  Stops: {row[f'{key_075_10}_stop_pct']:.1f}%, Targets: {row[f'{key_075_10}_target_pct']:.1f}%")

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Sharpe improvement heatmap
        ax = axes[0, 0]
        sharpe_improvements = pd.DataFrame()
        for stop, target in STOP_TARGET_PAIRS[:-1]:  # Exclude no stop/target
            key = f"stop_{stop}_target_{target}"
            if f"{key}_sharpe" in results_df.columns:
                sharpe_improvements[f"{stop}/{target}"] = results_df[f"{key}_sharpe"] - results_df['base_sharpe']

        if not sharpe_improvements.empty:
            avg_improvements = sharpe_improvements.mean()
            ax.bar(range(len(avg_improvements)), avg_improvements.values)
            ax.set_xticks(range(len(avg_improvements)))
            ax.set_xticklabels(avg_improvements.index, rotation=45)
            ax.set_ylabel('Average Sharpe Improvement')
            ax.set_title('Sharpe Improvement by Stop/Target Config')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)

        # 2. Return distribution
        ax = axes[0, 1]
        if f"stop_0.075_target_0.1_return" in results_df.columns:
            base_returns = results_df['base_return'] * 100
            modified_returns = results_df['stop_0.075_target_0.1_return'] * 100

            ax.scatter(base_returns, modified_returns, alpha=0.6)
            ax.plot([-50, 50], [-50, 50], 'r--', alpha=0.5)  # y=x line
            ax.set_xlabel('Base Return %')
            ax.set_ylabel('Return with 0.075/0.1 Stop/Target %')
            ax.set_title('Return Comparison')
            ax.grid(True, alpha=0.3)

        # 3. Exit type distribution
        ax = axes[1, 0]
        if f"stop_0.075_target_0.1_stop_pct" in results_df.columns:
            exit_data = pd.DataFrame({
                'Stops': results_df['stop_0.075_target_0.1_stop_pct'].values,
                'Targets': results_df['stop_0.075_target_0.1_target_pct'].values,
                'Signals': 100 - results_df['stop_0.075_target_0.1_stop_pct'].values - results_df['stop_0.075_target_0.1_target_pct'].values
            })

            exit_data.mean().plot(kind='bar', ax=ax)
            ax.set_ylabel('Percentage of Trades')
            ax.set_title('Average Exit Type Distribution (0.075/0.1 Config)')
            ax.set_xticklabels(['Stops', 'Targets', 'Signals'], rotation=0)

        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = "Summary Statistics:\n\n"
        summary_text += f"Strategies analyzed: {len(results_df)}\n"
        summary_text += f"Average trades/strategy: {results_df['num_trades'].mean():.0f}\n"
        summary_text += f"Average trades/day: {results_df['trades_per_day'].mean():.1f}\n\n"

        if '0.075/0.1' in config_df.index:
            summary_text += "0.075/0.1 Stop/Target Performance:\n"
            summary_text += f"  Avg Return: {config_df.loc['0.075/0.1', 'avg_return']:.2f}%\n"
            summary_text += f"  Avg Sharpe: {config_df.loc['0.075/0.1', 'avg_sharpe']:.2f}\n"
            summary_text += f"  Success Rate: {config_df.loc['0.075/0.1', 'win_rate']:.1f}%\n"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12, family='monospace')

        plt.tight_layout()
        plt.show()

        # Save results
        results_df.to_csv(run_dir / 'stop_target_analysis.csv', index=False)
        optimal_df.to_csv(run_dir / 'optimal_configurations.csv', index=False)

        print(f"\n✅ Analysis complete! Results saved to:")
        print(f"  - {run_dir}/stop_target_analysis.csv")
        print(f"  - {run_dir}/optimal_configurations.csv")

    else:
        print("\n❌ No valid strategies found for analysis")
else:
    print("❌ No performance data available")


# In[ ]:




