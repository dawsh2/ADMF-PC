# Shared helper functions for analysis snippets
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

def load_sql_query(filename: str, base_path: str = 'src/analytics/queries') -> str:
    """Load a SQL query from file."""
    filepath = Path(base_path) / filename
    if not filepath.suffix:
        filepath = filepath.with_suffix('.sql')
    
    with open(filepath, 'r') as f:
        return f.read()

def format_large_number(num: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(num) >= 1e9:
        return f"{num/1e9:.1f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.0f}"

def calculate_rolling_sharpe(returns: pd.Series, window: int = 252, 
                           periods_per_year: int = 252*78) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(periods_per_year)
    return rolling_sharpe

def get_strategy_metadata(strategy_hash: str, con) -> Dict:
    """Get metadata for a specific strategy."""
    query = f"""
    SELECT * FROM strategies 
    WHERE strategy_hash = '{strategy_hash}'
    LIMIT 1
    """
    result = con.execute(query).df()
    if not result.empty:
        return result.iloc[0].to_dict()
    return {}

def plot_strategy_comparison(strategies_df: pd.DataFrame, 
                           metric: str = 'sharpe_ratio',
                           top_n: int = 10):
    """Create a bar plot comparing strategies by a metric."""
    import matplotlib.pyplot as plt
    
    top_strategies = strategies_df.nlargest(top_n, metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(top_strategies)), top_strategies[metric])
    
    # Color bars by strategy type
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_strategies['strategy_type'].unique())))
    color_map = {stype: colors[i] for i, stype in enumerate(top_strategies['strategy_type'].unique())}
    
    for i, (idx, row) in enumerate(top_strategies.iterrows()):
        bars[i].set_color(color_map[row['strategy_type']])
    
    ax.set_xticks(range(len(top_strategies)))
    ax.set_xticklabels([f"{row['strategy_type'][:4]}..{row['strategy_hash'][-4:]}" 
                        for _, row in top_strategies.iterrows()], rotation=45)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Top {top_n} Strategies by {metric.replace("_", " ").title()}')
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for stype, color in color_map.items()]
    ax.legend(handles, color_map.keys(), title='Strategy Type', loc='upper right')
    
    plt.tight_layout()
    return fig, ax

def find_correlated_features(df: pd.DataFrame, target_col: str, 
                           threshold: float = 0.3) -> pd.DataFrame:
    """Find features correlated with a target column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
    
    # Filter by threshold
    significant = correlations[abs(correlations) > threshold]
    significant = significant[significant.index != target_col]
    
    return pd.DataFrame({
        'feature': significant.index,
        'correlation': significant.values,
        'abs_correlation': abs(significant.values)
    }).sort_values('abs_correlation', ascending=False)

def create_performance_summary(strategies_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table of performance metrics by strategy type."""
    summary = strategies_df.groupby('strategy_type').agg({
        'sharpe_ratio': ['count', 'mean', 'std', 'max'],
        'total_return': ['mean', 'std', 'max'],
        'max_drawdown': ['mean', 'min'],  # min because drawdowns are negative
        'win_rate': 'mean',
        'total_trades': 'mean'
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Sort by average Sharpe
    summary = summary.sort_values('sharpe_ratio_mean', ascending=False)
    
    return summary

def export_ensemble_config(ensemble_df: pd.DataFrame, 
                         output_path: str = 'ensemble_config.json') -> Dict:
    """Export ensemble configuration for production use."""
    import json
    
    config = {
        'ensemble_id': f"ensemble_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        'creation_date': pd.Timestamp.now().isoformat(),
        'strategies': [],
        'metrics': {
            'num_strategies': len(ensemble_df),
            'avg_sharpe': ensemble_df['sharpe_ratio'].mean(),
            'min_sharpe': ensemble_df['sharpe_ratio'].min(),
            'max_sharpe': ensemble_df['sharpe_ratio'].max(),
            'avg_return': ensemble_df['total_return'].mean()
        }
    }
    
    for _, strategy in ensemble_df.iterrows():
        config['strategies'].append({
            'strategy_hash': strategy['strategy_hash'],
            'strategy_type': strategy['strategy_type'],
            'weight': 1.0 / len(ensemble_df),  # Equal weight by default
            'sharpe_ratio': strategy['sharpe_ratio'],
            'total_return': strategy['total_return']
        })
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Ensemble configuration saved to {output_path}")
    return config

# Quick access functions for common queries
def get_top_sharpe(con, n: int = 10, min_trades: int = 20) -> pd.DataFrame:
    """Quick function to get top strategies by Sharpe ratio."""
    return con.execute(f"""
        SELECT * FROM strategies 
        WHERE total_trades >= {min_trades}
        ORDER BY sharpe_ratio DESC 
        LIMIT {n}
    """).df()

def get_signal_stats(con, strategy_hash: str) -> Dict:
    """Get signal statistics for a specific strategy."""
    stats = con.execute(f"""
        SELECT 
            COUNT(*) as total_signals,
            SUM(CASE WHEN val > 0 THEN 1 ELSE 0 END) as long_signals,
            SUM(CASE WHEN val < 0 THEN 1 ELSE 0 END) as short_signals,
            COUNT(DISTINCT DATE(ts)) as trading_days,
            MIN(ts) as first_signal,
            MAX(ts) as last_signal
        FROM signals
        WHERE strategy_hash = '{strategy_hash}' AND val != 0
    """).df()
    
    if not stats.empty:
        return stats.iloc[0].to_dict()
    return {}

def get_actual_trading_days(performance_df: pd.DataFrame, market_data: pd.DataFrame) -> int:
    """
    Calculate actual trading days based on when strategies were active.
    This avoids the bug of using all days in the market data.
    
    Returns:
        int: Number of actual trading days
    """
    # Method 1: If performance_df has trade timing info
    if 'first_signal_date' in performance_df.columns and 'last_signal_date' in performance_df.columns:
        # Use actual trading period from signals
        first_date = pd.to_datetime(performance_df['first_signal_date'].min())
        last_date = pd.to_datetime(performance_df['last_signal_date'].max())
        trading_days = len(pd.bdate_range(first_date, last_date))
        print(f"Trading period from signals: {first_date.date()} to {last_date.date()}")
        return trading_days
    
    # Method 2: Infer from trace files if available
    if 'trace_path' in performance_df.columns and len(performance_df) > 0:
        try:
            # Load first trace to check dates
            sample_trace = pd.read_parquet(performance_df.iloc[0]['trace_path'])
            if 'timestamp' in sample_trace.columns and len(sample_trace) > 0:
                first_date = sample_trace['timestamp'].min()
                last_date = sample_trace['timestamp'].max()
                trading_days = len(pd.bdate_range(first_date, last_date))
                print(f"Trading period from traces: {first_date.date()} to {last_date.date()}")
                return trading_days
        except Exception as e:
            print(f"Could not read trace file: {e}")
    
    # Method 3: If we have run_dir in globals, check for signal files
    if 'run_dir' in globals() and Path(globals()['run_dir']).exists():
        signal_files = list(Path(globals()['run_dir']).glob('signals/*.parquet'))
        if signal_files:
            try:
                # Read first signal file to get date range
                signals = pd.read_parquet(signal_files[0])
                if 'timestamp' in signals.columns and len(signals) > 0:
                    first_date = signals['timestamp'].min()
                    last_date = signals['timestamp'].max()
                    trading_days = len(pd.bdate_range(first_date, last_date))
                    print(f"Trading period from signal files: {first_date.date()} to {last_date.date()}")
                    return trading_days
            except Exception as e:
                print(f"Could not read signal file: {e}")
    
    # Method 4: Last resort - warn and use market data
    print("⚠️ WARNING: Could not determine actual trading period from signals/traces.")
    print("⚠️ Using full market data range - this may overstate trading days!")
    trading_days = len(market_data['timestamp'].dt.date.unique())
    date_range = f"{market_data['timestamp'].min().date()} to {market_data['timestamp'].max().date()}"
    print(f"Market data period: {date_range}")
    return trading_days

print("Helper functions loaded successfully!")
print("Available functions:")
print("  - load_sql_query(filename)")
print("  - format_large_number(num)")
print("  - calculate_rolling_sharpe(returns, window)")
print("  - get_strategy_metadata(strategy_hash, con)")
print("  - plot_strategy_comparison(df, metric)")
print("  - find_correlated_features(df, target_col)")
print("  - create_performance_summary(df)")
print("  - export_ensemble_config(ensemble_df)")
print("  - get_top_sharpe(con, n)")
print("  - get_signal_stats(con, strategy_hash)")
print("  - get_actual_trading_days(performance_df, market_data) # NEW!")