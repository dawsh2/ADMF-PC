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

def load_signal_data(file_path: Path) -> pd.DataFrame:
    """Load and prepare signal data from parquet file."""
    try:
        df = pd.read_parquet(file_path)
        print(f"\nLoaded {file_path.name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print("Sample data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_trade_metrics(df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """Calculate comprehensive trading metrics from signal data."""
    if df is None or df.empty:
        return {"error": "No data available"}
    
    print(f"\n=== Analyzing {strategy_name} ===")
    
    # Ensure we have the required columns
    required_cols = ['direction', 'price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return {"error": f"Missing required columns: {missing_cols}"}
    
    # Basic signal statistics
    total_signals = len(df)
    signal_counts = df['direction'].value_counts()
    
    print(f"Total signals: {total_signals}")
    print(f"Signal distribution: {signal_counts.to_dict()}")
    
    # Convert direction to numeric signals for analysis
    # Assuming: 1 = buy, -1 = sell, 0 = flat/hold
    direction_map = {'buy': 1, 'sell': -1, 'flat': 0, 'hold': 0}
    df_analysis = df.copy()
    
    # Try to map directions, handle various formats
    if df_analysis['direction'].dtype == 'object':
        # Try direct mapping first
        df_analysis['signal'] = df_analysis['direction'].map(direction_map)
        
        # If mapping failed, try lowercase
        if df_analysis['signal'].isna().all():
            df_analysis['signal'] = df_analysis['direction'].str.lower().map(direction_map)
        
        # If still failed, try numeric conversion
        if df_analysis['signal'].isna().all():
            try:
                df_analysis['signal'] = pd.to_numeric(df_analysis['direction'], errors='coerce')
            except:
                pass
    else:
        # Already numeric
        df_analysis['signal'] = df_analysis['direction']
    
    # Check if we successfully converted signals
    if df_analysis['signal'].isna().all():
        print(f"Could not parse direction values: {df['direction'].unique()[:10]}")
        return {"error": "Could not parse direction values"}
    
    # Remove NaN signals
    df_analysis = df_analysis.dropna(subset=['signal', 'price'])
    
    if df_analysis.empty:
        return {"error": "No valid signal data after cleaning"}
    
    print(f"Valid signals after cleaning: {len(df_analysis)}")
    
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
    
    # Calculate cumulative returns
    df_analysis['cumulative_return'] = (1 + df_analysis['strategy_return']).cumprod()
    
    # Basic performance metrics
    total_return = df_analysis['cumulative_return'].iloc[-1] - 1
    
    # Trading activity metrics
    signal_changes = (df_analysis['signal'] != df_analysis['signal'].shift(1)).sum()
    non_zero_signals = (df_analysis['signal'] != 0).sum()
    
    # Win/Loss analysis
    positive_returns = df_analysis[df_analysis['strategy_return'] > 0]
    negative_returns = df_analysis[df_analysis['strategy_return'] < 0]
    zero_returns = df_analysis[df_analysis['strategy_return'] == 0]
    
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
    
    metrics = {
        'strategy_name': strategy_name,
        'total_signals': total_signals,
        'valid_signals': len(df_analysis),
        'signal_distribution': signal_counts.to_dict(),
        'time_span_days': time_span.total_seconds() / (24 * 3600),
        'total_return_pct': total_return * 100,
        'signal_changes': signal_changes,
        'non_zero_signals': non_zero_signals,
        'total_trades': total_trades,
        'winning_trades': len(positive_returns),
        'losing_trades': len(negative_returns),
        'flat_periods': len(zero_returns),
        'win_rate_pct': win_rate * 100,
        'avg_win_pct': avg_win * 100,
        'avg_loss_pct': avg_loss * 100,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown * 100,
        'volatility_annualized_pct': annualized_volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'annualized_return_pct': annualized_return * 100
    }
    
    return metrics

def print_performance_summary(metrics: Dict[str, Any]):
    """Print formatted performance summary."""
    if 'error' in metrics:
        print(f"Error: {metrics['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY: {metrics['strategy_name']}")
    print(f"{'='*60}")
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Total Signals: {metrics['total_signals']:,}")
    print(f"  Valid Signals: {metrics['valid_signals']:,}")
    print(f"  Time Span: {metrics['time_span_days']:.1f} days")
    print(f"  Signal Changes: {metrics['signal_changes']:,}")
    print(f"  Non-Zero Signals: {metrics['non_zero_signals']:,}")
    
    print(f"\n📈 SIGNAL DISTRIBUTION:")
    for signal_type, count in metrics['signal_distribution'].items():
        print(f"  {signal_type}: {count:,} ({count/metrics['total_signals']*100:.1f}%)")
    
    print(f"\n💰 PERFORMANCE METRICS:")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {metrics['annualized_return_pct']:.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Volatility (Ann.): {metrics['volatility_annualized_pct']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print(f"\n🎯 TRADING METRICS:")
    print(f"  Total Trades: {metrics['total_trades']:,}")
    print(f"  Winning Trades: {metrics['winning_trades']:,}")
    print(f"  Losing Trades: {metrics['losing_trades']:,}")
    print(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
    print(f"  Average Win: {metrics['avg_win_pct']:.3f}%")
    print(f"  Average Loss: {metrics['avg_loss_pct']:.3f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

def compare_strategies(metrics1: Dict, metrics2: Dict):
    """Compare two strategies side by side."""
    if 'error' in metrics1 or 'error' in metrics2:
        print("Cannot compare strategies due to errors in data")
        return
    
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    comparison_metrics = [
        ('Total Return %', 'total_return_pct', '.2f'),
        ('Annualized Return %', 'annualized_return_pct', '.2f'),
        ('Max Drawdown %', 'max_drawdown_pct', '.2f'),
        ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
        ('Win Rate %', 'win_rate_pct', '.1f'),
        ('Profit Factor', 'profit_factor', '.2f'),
        ('Total Trades', 'total_trades', ','),
        ('Volatility % (Ann.)', 'volatility_annualized_pct', '.2f')
    ]
    
    print(f"{'Metric':<25} {'Default':<15} {'Custom':<15} {'Difference':<15}")
    print("-" * 70)
    
    for metric_name, key, fmt in comparison_metrics:
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)
        diff = val2 - val1
        
        print(f"{metric_name:<25} {val1:{fmt}:<15} {val2:{fmt}:<15} {diff:+{fmt}:<15}")

def main():
    """Main analysis function."""
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9")
    signals_path = workspace_path / "traces" / "SPY_1m" / "signals" / "unknown"
    
    # File paths
    default_file = signals_path / "SPY_adaptive_ensemble_default.parquet"
    custom_file = signals_path / "SPY_adaptive_ensemble_custom.parquet"
    
    print("🔍 Loading signal trace files...")
    
    # Load data
    default_data = load_signal_data(default_file)
    custom_data = load_signal_data(custom_file)
    
    # Calculate metrics
    print("\n📊 Calculating performance metrics...")
    default_metrics = calculate_trade_metrics(default_data, "Adaptive Ensemble Default")
    custom_metrics = calculate_trade_metrics(custom_data, "Adaptive Ensemble Custom")
    
    # Print individual summaries
    print_performance_summary(default_metrics)
    print_performance_summary(custom_metrics)
    
    # Compare strategies
    compare_strategies(default_metrics, custom_metrics)
    
    # Save results to CSV
    if 'error' not in default_metrics and 'error' not in custom_metrics:
        results_df = pd.DataFrame([default_metrics, custom_metrics])
        output_file = workspace_path / "ensemble_performance_comparison.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()