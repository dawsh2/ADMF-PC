#!/usr/bin/env python3
"""
Calculate Sharpe ratio for the ensemble strategy.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def calculate_sharpe_ratio(df, risk_free_rate=0.05):
    """
    Calculate annualized Sharpe ratio from signal data.
    
    Args:
        df: DataFrame with signal changes and prices
        risk_free_rate: Annual risk-free rate (default 5%)
    """
    if df.empty:
        return None
    
    # Reconstruct full price series and strategy returns
    df = df.sort_values('bar_idx').reset_index(drop=True)
    
    # Create a full timeline from min to max bar index
    min_idx = df['bar_idx'].min()
    max_idx = df['bar_idx'].max()
    full_timeline = pd.DataFrame({'bar_idx': range(min_idx, max_idx + 1)})
    
    # Merge with signal data and forward fill signals
    timeline_with_signals = full_timeline.merge(df[['bar_idx', 'signal_value', 'price']], 
                                                on='bar_idx', how='left')
    
    # Forward fill signals (carry forward last signal)
    timeline_with_signals['signal_value'] = timeline_with_signals['signal_value'].fillna(method='ffill').fillna(0)
    
    # Interpolate prices for missing bars (linear interpolation)
    timeline_with_signals['price'] = timeline_with_signals['price'].interpolate()
    
    # Remove any remaining NaN rows
    timeline_with_signals = timeline_with_signals.dropna()
    
    if len(timeline_with_signals) < 2:
        return None
    
    # Calculate price returns
    timeline_with_signals['price_return'] = timeline_with_signals['price'].pct_change()
    
    # Calculate strategy returns (signal * price return)
    timeline_with_signals['strategy_return'] = (
        timeline_with_signals['signal_value'].shift(1) * timeline_with_signals['price_return']
    )
    
    # Remove first row (NaN from pct_change and shift)
    strategy_returns = timeline_with_signals['strategy_return'].dropna()
    
    if len(strategy_returns) == 0:
        return None
    
    # Convert to daily returns (assuming 390 bars per trading day)
    bars_per_day = 390
    num_full_days = len(strategy_returns) // bars_per_day
    
    if num_full_days < 2:
        # Not enough data for daily analysis, use bar-level
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        
        # Annualize assuming 390 bars per day, 252 trading days
        annual_return = mean_return * bars_per_day * 252
        annual_volatility = std_return * np.sqrt(bars_per_day * 252)
        
        if annual_volatility == 0:
            return None
            
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        
        return {
            'sharpe_ratio': sharpe,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'data_points': len(strategy_returns),
            'method': 'bar_level'
        }
    
    else:
        # Aggregate to daily returns
        timeline_with_signals['day'] = timeline_with_signals.index // bars_per_day
        daily_returns = timeline_with_signals.groupby('day')['strategy_return'].sum()
        
        # Remove incomplete days
        daily_returns = daily_returns.iloc[:num_full_days]
        
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        
        # Annualize
        annual_return = mean_daily_return * 252
        annual_volatility = std_daily_return * np.sqrt(252)
        
        if annual_volatility == 0:
            return None
            
        sharpe = (annual_return - risk_free_rate) / annual_volatility
        
        return {
            'sharpe_ratio': sharpe,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'daily_returns_count': len(daily_returns),
            'method': 'daily_aggregation'
        }

def analyze_strategy_performance(signal_file):
    """Analyze strategy performance including Sharpe ratio."""
    
    print(f"Reading signal file: {signal_file}")
    df = pd.read_parquet(signal_file)
    
    # Rename columns
    df = df.rename(columns={
        'idx': 'bar_idx',
        'px': 'price', 
        'val': 'signal_value'
    })
    
    print(f"Loaded {len(df)} signal records")
    print(f"Bar range: {df['bar_idx'].min()} to {df['bar_idx'].max()}")
    print(f"Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
    
    # Calculate Sharpe ratio
    sharpe_results = calculate_sharpe_ratio(df)
    
    if sharpe_results is None:
        print("Could not calculate Sharpe ratio - insufficient data")
        return
    
    print(f"\n{'='*60}")
    print("SHARPE RATIO ANALYSIS")
    print(f"{'='*60}")
    print(f"Method: {sharpe_results['method']}")
    print(f"Data points: {sharpe_results.get('data_points', sharpe_results.get('daily_returns_count', 'N/A'))}")
    print(f"Annualized return: {sharpe_results['annual_return']:.2%}")
    print(f"Annualized volatility: {sharpe_results['annual_volatility']:.2%}")
    print(f"Sharpe ratio: {sharpe_results['sharpe_ratio']:.2f}")
    
    # Categorize Sharpe ratio
    sharpe = sharpe_results['sharpe_ratio']
    if sharpe > 2.0:
        category = "Excellent (>2.0)"
    elif sharpe > 1.0:
        category = "Good (1.0-2.0)"
    elif sharpe > 0.5:
        category = "Acceptable (0.5-1.0)"
    elif sharpe > 0:
        category = "Poor (0-0.5)"
    else:
        category = "Negative (<0)"
    
    print(f"Sharpe category: {category}")
    
    # Also calculate without transaction costs for comparison
    print(f"\nNote: This is gross Sharpe (before transaction costs)")
    print(f"With 0.5bp costs, expect Sharpe to be ~0.2-0.3 lower")

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_sharpe.py <parquet_file_path>")
        sys.exit(1)
    
    signal_file = Path(sys.argv[1])
    if not signal_file.exists():
        print(f"File not found: {signal_file}")
        sys.exit(1)
    
    analyze_strategy_performance(signal_file)

if __name__ == "__main__":
    main()