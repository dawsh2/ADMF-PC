#!/usr/bin/env python3
"""
Analyze performance metrics for adaptive ensemble strategies from signal trace files.

This script reads the parquet files for both adaptive ensemble strategies and calculates
key performance metrics including returns, trades, win rate, drawdown, and Sharpe ratio.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def load_price_data() -> pd.DataFrame:
    """Load SPY price data."""
    try:
        price_df = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_1m.csv')
        # Convert timestamp to datetime and set as index
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df = price_df.set_index('timestamp')
        print(f"Loaded price data: {len(price_df)} bars from {price_df.index.min()} to {price_df.index.max()}")
        print(f"Price columns: {list(price_df.columns)}")
        return price_df
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None

def load_signal_data(file_path: Path) -> pd.DataFrame:
    """Load and prepare signal data from parquet file."""
    try:
        df = pd.read_parquet(file_path)
        print(f"\nLoaded {file_path.name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Convert timestamps and set as index
        df['timestamp'] = pd.to_datetime(df['ts'])
        df = df.set_index('timestamp').sort_index()
        
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Signal values: {df['val'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_trade_metrics(df: pd.DataFrame, price_df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """Calculate comprehensive trading metrics from signal data."""
    if df is None or df.empty:
        return {"error": "No signal data available"}
    
    if price_df is None or price_df.empty:
        return {"error": "No price data available"}
    
    print(f"\n=== Analyzing {strategy_name} ===")
    
    # Merge signal data with price data
    df_merged = df.join(price_df[['close']], how='inner')
    
    if df_merged.empty:
        print("No matching timestamps between signals and price data")
        return {"error": "No matching timestamps between signals and price data"}
    
    print(f"Merged data: {len(df_merged)} rows with both signals and prices")
    
    # Use the merged data for analysis
    df_analysis = df_merged.copy()
    df_analysis['signal'] = df_analysis['val']  # Use original numeric values
    df_analysis['price'] = df_analysis['close'] # Use close price
    
    # Map signal values to directions for display
    signal_map = {1: 'buy', -1: 'sell', 0: 'flat'}
    df_analysis['direction'] = df_analysis['signal'].map(signal_map)
    
    # Basic signal statistics
    total_signals = len(df_analysis)
    signal_counts = df_analysis['direction'].value_counts()
    
    print(f"Total signals: {total_signals}")
    print(f"Signal distribution: {signal_counts.to_dict()}")
    
    # Calculate price changes and returns
    df_analysis = df_analysis.sort_index()
    df_analysis['price_change'] = df_analysis['price'].pct_change()
    df_analysis['forward_return'] = df_analysis['price'].pct_change().shift(-1)
    
    # Calculate strategy returns (signal * forward return)
    df_analysis['strategy_return'] = df_analysis['signal'] * df_analysis['forward_return']
    
    # Remove first and last rows due to NaN from pct_change and shift
    df_analysis = df_analysis.dropna(subset=['strategy_return'])
    
    if df_analysis.empty:
        return {"error": "No valid returns after calculation"}
    
    print(f"Valid returns: {len(df_analysis)}")
    
    # Calculate cumulative returns
    df_analysis['cumulative_return'] = (1 + df_analysis['strategy_return']).cumprod()
    
    # Basic performance metrics
    total_return = df_analysis['cumulative_return'].iloc[-1] - 1
    
    # Trading activity metrics
    signal_changes = (df_analysis['signal'] != df_analysis['signal'].shift(1)).sum()
    non_zero_signals = (df_analysis['signal'] != 0).sum()
    active_periods = non_zero_signals / len(df_analysis)
    
    # Win/Loss analysis on non-zero signals only
    active_trades = df_analysis[df_analysis['signal'] != 0]
    if len(active_trades) == 0:
        return {"error": "No active trading signals found"}
    
    positive_returns = active_trades[active_trades['strategy_return'] > 0]
    negative_returns = active_trades[active_trades['strategy_return'] < 0]
    zero_returns = active_trades[active_trades['strategy_return'] == 0]
    
    total_trades = len(positive_returns) + len(negative_returns)
    win_rate = len(positive_returns) / total_trades if total_trades > 0 else 0
    
    avg_win = positive_returns['strategy_return'].mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns['strategy_return'].mean() if len(negative_returns) > 0 else 0
    
    profit_factor = abs(avg_win * len(positive_returns) / (avg_loss * len(negative_returns))) if avg_loss != 0 and len(negative_returns) > 0 else float('inf')
    
    # Drawdown calculation
    running_max = df_analysis['cumulative_return'].expanding().max()
    drawdown = (df_analysis['cumulative_return'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Volatility and Sharpe ratio (annualized, assuming 1-minute data)
    returns_std = df_analysis['strategy_return'].std()
    returns_mean = df_analysis['strategy_return'].mean()
    
    # Annualize assuming 252 trading days * 6.5 hours * 60 minutes
    periods_per_year = 252 * 6.5 * 60
    annualized_return = (1 + returns_mean) ** periods_per_year - 1
    annualized_volatility = returns_std * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Time-based metrics
    time_span = df_analysis.index.max() - df_analysis.index.min()
    avg_trade_duration = calculate_avg_trade_duration(df_analysis)
    
    # Calculate some additional metrics
    max_consecutive_wins = calculate_consecutive_runs(positive_returns.index, df_analysis.index)
    max_consecutive_losses = calculate_consecutive_runs(negative_returns.index, df_analysis.index)
    
    metrics = {
        'strategy_name': strategy_name,
        'total_signals': total_signals,
        'valid_signals': len(df_analysis),
        'signal_distribution': signal_counts.to_dict(),
        'time_span_days': time_span.total_seconds() / (24 * 3600),
        'total_return_pct': total_return * 100,
        'signal_changes': signal_changes,
        'non_zero_signals': non_zero_signals,
        'active_periods_pct': active_periods * 100,
        'total_active_trades': len(active_trades),
        'total_trades': total_trades,
        'winning_trades': len(positive_returns),
        'losing_trades': len(negative_returns),
        'flat_trades': len(zero_returns),
        'win_rate_pct': win_rate * 100,
        'avg_win_pct': avg_win * 100,
        'avg_loss_pct': avg_loss * 100,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown * 100,
        'volatility_annualized_pct': annualized_volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'annualized_return_pct': annualized_return * 100,
        'avg_trade_duration_minutes': avg_trade_duration,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }
    
    return metrics

def calculate_avg_trade_duration(df: pd.DataFrame) -> float:
    """Calculate average trade duration in minutes."""
    try:
        # Find trade segments (continuous non-zero signals)
        df = df[df['signal'] != 0].copy()
        if len(df) < 2:
            return 0
        
        # Calculate time differences in minutes
        time_diffs = df.index.to_series().diff().dt.total_seconds() / 60
        # Filter out gaps > 5 minutes (likely trade boundaries)
        trade_durations = time_diffs[time_diffs <= 5]
        
        return trade_durations.mean() if len(trade_durations) > 0 else 0
    except:
        return 0

def calculate_consecutive_runs(trade_indices, all_indices) -> int:
    """Calculate maximum consecutive wins or losses."""
    try:
        if len(trade_indices) == 0:
            return 0
        
        # Find consecutive runs
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(trade_indices)):
            # Check if this trade follows immediately after the previous
            prev_idx = all_indices.get_loc(trade_indices[i-1])
            curr_idx = all_indices.get_loc(trade_indices[i])
            
            if curr_idx == prev_idx + 1:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        
        return max_consecutive
    except:
        return 0\n\ndef print_performance_summary(metrics: Dict[str, Any]):\n    \"\"\"Print formatted performance summary.\"\"\"\n    if 'error' in metrics:\n        print(f\"Error: {metrics['error']}\")\n        return\n    \n    print(f\"\\n{'='*60}\")\n    print(f\"PERFORMANCE SUMMARY: {metrics['strategy_name']}\")\n    print(f\"{'='*60}\")\n    \n    print(f\"\\n📊 BASIC STATISTICS:\")\n    print(f\"  Total Signals: {metrics['total_signals']:,}\")\n    print(f\"  Valid Signals: {metrics['valid_signals']:,}\")\n    print(f\"  Time Span: {metrics['time_span_days']:.1f} days\")\n    print(f\"  Signal Changes: {metrics['signal_changes']:,}\")\n    print(f\"  Active Periods: {metrics['active_periods_pct']:.1f}%\")\n    \n    print(f\"\\n📈 SIGNAL DISTRIBUTION:\")\n    for signal_type, count in metrics['signal_distribution'].items():\n        pct = count/metrics['total_signals']*100\n        print(f\"  {signal_type}: {count:,} ({pct:.1f}%)\")\n    \n    print(f\"\\n💰 PERFORMANCE METRICS:\")\n    print(f\"  Total Return: {metrics['total_return_pct']:.2f}%\")\n    print(f\"  Annualized Return: {metrics['annualized_return_pct']:.2f}%\")\n    print(f\"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\")\n    print(f\"  Volatility (Ann.): {metrics['volatility_annualized_pct']:.2f}%\")\n    print(f\"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\")\n    \n    print(f\"\\n🎯 TRADING METRICS:\")\n    print(f\"  Total Active Trades: {metrics['total_active_trades']:,}\")\n    print(f\"  Profitable Trades: {metrics['winning_trades']:,}\")\n    print(f\"  Losing Trades: {metrics['losing_trades']:,}\")\n    print(f\"  Flat Trades: {metrics['flat_trades']:,}\")\n    print(f\"  Win Rate: {metrics['win_rate_pct']:.1f}%\")\n    print(f\"  Average Win: {metrics['avg_win_pct']:.3f}%\")\n    print(f\"  Average Loss: {metrics['avg_loss_pct']:.3f}%\")\n    print(f\"  Profit Factor: {metrics['profit_factor']:.2f}\")\n    \n    print(f\"\\n⏱️ TIMING METRICS:\")\n    print(f\"  Avg Trade Duration: {metrics['avg_trade_duration_minutes']:.1f} minutes\")\n    print(f\"  Max Consecutive Wins: {metrics['max_consecutive_wins']}\")\n    print(f\"  Max Consecutive Losses: {metrics['max_consecutive_losses']}\")\n\ndef compare_strategies(metrics1: Dict, metrics2: Dict):\n    \"\"\"Compare two strategies side by side.\"\"\"\n    if 'error' in metrics1 or 'error' in metrics2:\n        print(\"Cannot compare strategies due to errors in data\")\n        return\n    \n    print(f\"\\n{'='*80}\")\n    print(f\"STRATEGY COMPARISON\")\n    print(f\"{'='*80}\")\n    \n    comparison_metrics = [\n        ('Total Return %', 'total_return_pct', '.2f'),\n        ('Annualized Return %', 'annualized_return_pct', '.2f'),\n        ('Max Drawdown %', 'max_drawdown_pct', '.2f'),\n        ('Sharpe Ratio', 'sharpe_ratio', '.2f'),\n        ('Win Rate %', 'win_rate_pct', '.1f'),\n        ('Profit Factor', 'profit_factor', '.2f'),\n        ('Total Active Trades', 'total_active_trades', ','),\n        ('Volatility % (Ann.)', 'volatility_annualized_pct', '.2f'),\n        ('Active Periods %', 'active_periods_pct', '.1f'),\n        ('Avg Trade Duration (min)', 'avg_trade_duration_minutes', '.1f')\n    ]\n    \n    print(f\"{'Metric':<25} {'Default':<15} {'Custom':<15} {'Difference':<15}\")\n    print(\"-\" * 70)\n    \n    for metric_name, key, fmt in comparison_metrics:\n        val1 = metrics1.get(key, 0)\n        val2 = metrics2.get(key, 0)\n        diff = val2 - val1\n        \n        print(f\"{metric_name:<25} {val1:{fmt}:<15} {val2:{fmt}:<15} {diff:+{fmt}:<15}\")\n\ndef main():\n    \"\"\"Main analysis function.\"\"\"\n    workspace_path = Path(\"/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9\")\n    signals_path = workspace_path / \"traces\" / \"SPY_1m\" / \"signals\" / \"unknown\"\n    \n    # File paths\n    default_file = signals_path / \"SPY_adaptive_ensemble_default.parquet\"\n    custom_file = signals_path / \"SPY_adaptive_ensemble_custom.parquet\"\n    \n    print(\"🔍 Loading data files...\")\n    \n    # Load price data\n    print(\"\\n📈 Loading price data...\")\n    price_data = load_price_data()\n    \n    if price_data is None:\n        print(\"❌ Cannot proceed without price data\")\n        return\n    \n    # Load signal data\n    print(\"\\n📊 Loading signal data...\")\n    default_data = load_signal_data(default_file)\n    custom_data = load_signal_data(custom_file)\n    \n    # Calculate metrics\n    print(\"\\n⚡ Calculating performance metrics...\")\n    default_metrics = calculate_trade_metrics(default_data, price_data, \"Adaptive Ensemble Default\")\n    custom_metrics = calculate_trade_metrics(custom_data, price_data, \"Adaptive Ensemble Custom\")\n    \n    # Print individual summaries\n    print_performance_summary(default_metrics)\n    print_performance_summary(custom_metrics)\n    \n    # Compare strategies\n    compare_strategies(default_metrics, custom_metrics)\n    \n    # Save results to CSV\n    if 'error' not in default_metrics and 'error' not in custom_metrics:\n        results_df = pd.DataFrame([default_metrics, custom_metrics])\n        output_file = workspace_path / \"ensemble_performance_comparison.csv\"\n        results_df.to_csv(output_file, index=False)\n        print(f\"\\n💾 Results saved to: {output_file}\")\n    \n    print(f\"\\n✅ Analysis complete!\")\n\nif __name__ == \"__main__\":\n    main()