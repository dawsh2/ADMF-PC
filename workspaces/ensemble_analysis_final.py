#!/usr/bin/env python3
"""
Analyze performance metrics for adaptive ensemble strategies from signal trace files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

def load_price_data() -> pd.DataFrame:
    """Load SPY price data."""
    try:
        price_df = pd.read_csv('/Users/daws/ADMF-PC/data/SPY_1m.csv')
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df = price_df.set_index('timestamp')
        print(f"📊 Loaded price data: {len(price_df)} bars from {price_df.index.min()} to {price_df.index.max()}")
        return price_df
    except Exception as e:
        print(f"❌ Error loading price data: {e}")
        return None

def load_signal_data(file_path: Path) -> pd.DataFrame:
    """Load and prepare signal data from parquet file."""
    try:
        df = pd.read_parquet(file_path)
        print(f"📈 Loaded {file_path.name}: {df.shape[0]} signals")
        
        # Convert timestamps and set as index
        df['timestamp'] = pd.to_datetime(df['ts'])
        df = df.set_index('timestamp').sort_index()
        
        signal_counts = df['val'].value_counts()
        print(f"   Signal distribution: {signal_counts.to_dict()}")
        
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def calculate_metrics(signal_df: pd.DataFrame, price_df: pd.DataFrame, strategy_name: str) -> Dict[str, Any]:
    """Calculate comprehensive trading metrics."""
    if signal_df is None or price_df is None:
        return {"error": "Missing data"}
    
    print(f"\n⚡ Analyzing {strategy_name}...")
    
    # Merge signal data with price data
    merged = signal_df.join(price_df[['Close']], how='inner')
    
    if merged.empty:
        return {"error": "No matching timestamps"}
    
    print(f"   Merged data: {len(merged)} rows")
    
    # Prepare data
    merged['signal'] = merged['val']  # -1, 0, 1
    merged['price'] = merged['Close']
    
    # Calculate returns
    merged = merged.sort_index()
    merged['price_change'] = merged['price'].pct_change()
    merged['forward_return'] = merged['price'].pct_change().shift(-1)
    merged['strategy_return'] = merged['signal'] * merged['forward_return']
    
    # Clean data
    valid_data = merged.dropna(subset=['strategy_return'])
    
    if valid_data.empty:
        return {"error": "No valid returns"}
    
    # Calculate cumulative returns
    valid_data['cumulative_return'] = (1 + valid_data['strategy_return']).cumprod()
    
    # Basic stats
    total_return = valid_data['cumulative_return'].iloc[-1] - 1
    
    # Active trading analysis
    active_trades = valid_data[valid_data['signal'] != 0]
    
    if len(active_trades) == 0:
        return {"error": "No active trades"}
    
    # Win/Loss analysis
    positive_trades = active_trades[active_trades['strategy_return'] > 0]
    negative_trades = active_trades[active_trades['strategy_return'] < 0]
    
    total_active = len(positive_trades) + len(negative_trades)
    win_rate = len(positive_trades) / total_active if total_active > 0 else 0
    
    avg_win = positive_trades['strategy_return'].mean() if len(positive_trades) > 0 else 0
    avg_loss = negative_trades['strategy_return'].mean() if len(negative_trades) > 0 else 0
    
    profit_factor = abs(avg_win * len(positive_trades) / (avg_loss * len(negative_trades))) if avg_loss < 0 and len(negative_trades) > 0 else float('inf')
    
    # Drawdown
    running_max = valid_data['cumulative_return'].expanding().max()
    drawdown = (valid_data['cumulative_return'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Risk metrics
    returns_std = valid_data['strategy_return'].std()
    returns_mean = valid_data['strategy_return'].mean()
    
    # Annualize (252 days * 6.5 hours * 60 minutes)
    periods_per_year = 252 * 6.5 * 60
    annualized_return = (1 + returns_mean) ** periods_per_year - 1
    annualized_volatility = returns_std * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Time metrics
    time_span = valid_data.index.max() - valid_data.index.min()
    
    # Signal distribution
    signal_map = {1: 'buy', -1: 'sell', 0: 'flat'}
    signal_counts = {signal_map[k]: v for k, v in valid_data['signal'].value_counts().items()}
    
    return {
        'strategy_name': strategy_name,
        'total_signals': len(valid_data),
        'time_span_days': time_span.total_seconds() / (24 * 3600),
        'total_return_pct': total_return * 100,
        'annualized_return_pct': annualized_return * 100,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe_ratio,
        'volatility_annualized_pct': annualized_volatility * 100,
        'total_active_trades': len(active_trades),
        'winning_trades': len(positive_trades),
        'losing_trades': len(negative_trades),
        'win_rate_pct': win_rate * 100,
        'avg_win_pct': avg_win * 100,
        'avg_loss_pct': avg_loss * 100,
        'profit_factor': profit_factor,
        'active_periods_pct': len(active_trades) / len(valid_data) * 100,
        'signal_distribution': signal_counts
    }

def print_summary(metrics: Dict[str, Any]):
    """Print performance summary."""
    if 'error' in metrics:
        print(f"❌ Error: {metrics['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY: {metrics['strategy_name']}")
    print(f"{'='*60}")
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Total Signals: {metrics['total_signals']:,}")
    print(f"  Time Span: {metrics['time_span_days']:.1f} days")
    print(f"  Active Periods: {metrics['active_periods_pct']:.1f}%")
    
    print(f"\n📈 SIGNAL DISTRIBUTION:")
    for signal_type, count in metrics['signal_distribution'].items():
        pct = count/metrics['total_signals']*100
        print(f"  {signal_type}: {count:,} ({pct:.1f}%)")
    
    print(f"\n💰 PERFORMANCE METRICS:")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {metrics['annualized_return_pct']:.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Volatility (Ann.): {metrics['volatility_annualized_pct']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print(f"\n🎯 TRADING METRICS:")
    print(f"  Total Active Trades: {metrics['total_active_trades']:,}")
    print(f"  Winning Trades: {metrics['winning_trades']:,}")
    print(f"  Losing Trades: {metrics['losing_trades']:,}")
    print(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
    print(f"  Average Win: {metrics['avg_win_pct']:.3f}%")
    print(f"  Average Loss: {metrics['avg_loss_pct']:.3f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")

def compare_strategies(metrics1: Dict, metrics2: Dict):
    """Compare two strategies."""
    if 'error' in metrics1 or 'error' in metrics2:
        print("❌ Cannot compare strategies due to errors")
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
        ('Total Active Trades', 'total_active_trades', ','),
        ('Volatility % (Ann.)', 'volatility_annualized_pct', '.2f'),
        ('Active Periods %', 'active_periods_pct', '.1f')
    ]
    
    print(f"{'Metric':<25} {'Default':<15} {'Custom':<15} {'Difference':<15}")
    print("-" * 70)
    
    for metric_name, key, fmt in comparison_metrics:
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)
        diff = val2 - val1
        
        if fmt == ',':
            val1_str = f"{val1:,}"
            val2_str = f"{val2:,}"
            diff_str = f"{diff:+,}"
        elif fmt == '.2f':
            val1_str = f"{val1:.2f}"
            val2_str = f"{val2:.2f}"
            diff_str = f"{diff:+.2f}"
        elif fmt == '.1f':
            val1_str = f"{val1:.1f}"
            val2_str = f"{val2:.1f}"
            diff_str = f"{diff:+.1f}"
        else:
            val1_str = str(val1)
            val2_str = str(val2)
            diff_str = f"{diff:+}"
            
        print(f"{metric_name:<25} {val1_str:<15} {val2_str:<15} {diff_str:<15}")

def main():
    """Main analysis function."""
    workspace_path = Path("/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9")
    signals_path = workspace_path / "traces" / "SPY_1m" / "signals" / "unknown"
    
    default_file = signals_path / "SPY_adaptive_ensemble_default.parquet"
    custom_file = signals_path / "SPY_adaptive_ensemble_custom.parquet"
    
    print("🔍 Starting Ensemble Performance Analysis")
    print("="*50)
    
    # Load data
    price_data = load_price_data()
    if price_data is None:
        return
    
    default_data = load_signal_data(default_file)
    custom_data = load_signal_data(custom_file)
    
    # Calculate metrics
    print("\n⚡ Calculating performance metrics...")
    default_metrics = calculate_metrics(default_data, price_data, "Adaptive Ensemble Default")
    custom_metrics = calculate_metrics(custom_data, price_data, "Adaptive Ensemble Custom")
    
    # Display results
    print_summary(default_metrics)
    print_summary(custom_metrics)
    compare_strategies(default_metrics, custom_metrics)
    
    # Save results
    if 'error' not in default_metrics and 'error' not in custom_metrics:
        results_df = pd.DataFrame([default_metrics, custom_metrics])
        output_file = workspace_path / "ensemble_performance_comparison.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()