#!/usr/bin/env python3
"""
Regime Analysis Script

Analyzes classifier regime data and correlates with market movements
to determine optimal strategy parameters for each regime.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_regime_data(filepath: str) -> Tuple[pd.DataFrame, Dict]:
    """Load regime change data from sparse storage."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract regime changes
    changes = []
    for i, change in enumerate(data['changes']):
        # The 'idx' field represents the bar number
        # We'll need to map this to actual market timestamps later
        changes.append({
            'bar_idx': change['idx'],
            'timestamp': pd.to_datetime(change['ts']).tz_localize(None),
            'symbol': change['sym'],
            'regime': change['val'],
            'strategy': change['strat']
        })
    
    df = pd.DataFrame(changes)
    metadata = data['metadata']
    
    return df, metadata


def load_market_data(symbol: str = "SPY", start_date: str = None) -> pd.DataFrame:
    """Load market data for the analysis period."""
    # Try different data file formats - prioritize 1h data
    data_files = [
        Path(f"./data/{symbol}_1h.parquet"),
        Path(f"./data/{symbol}_1h.csv"),
        Path(f"./data/{symbol}_1m.csv"),  # Will resample if needed
        Path(f"./data/{symbol}.csv")      # Last resort
    ]
    
    for data_path in data_files:
        if data_path.exists():
            print(f"Loading market data from: {data_path}")
            
            if data_path.suffix == '.parquet':
                df = pd.read_parquet(data_path)
            else:
                # Try to infer the date column
                df = pd.read_csv(data_path)
                
                # Find datetime column
                date_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if date_cols:
                    df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], utc=True)
                    df.set_index(date_cols[0], inplace=True)
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    df.set_index('Date', inplace=True)
                else:
                    # Assume first column is date
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True)
                    df.set_index(df.columns[0], inplace=True)
            
            # Remove timezone info for compatibility
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Ensure we have OHLCV columns
            df.columns = df.columns.str.lower()
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required_cols):
                # Don't resample - we're using 1m data now
                # if '1m' in str(data_path):
                #     print("Resampling 1m data to 1h...")
                #     df = df.resample('1H').agg({
                #         'open': 'first',
                #         'high': 'max',
                #         'low': 'min',
                #         'close': 'last',
                #         'volume': 'sum'
                #     }).dropna()
                
                if start_date:
                    df = df[df.index >= start_date]
                
                print(f"Loaded {len(df)} bars of market data")
                return df
    
    print(f"No data file found for {symbol}")
    return pd.DataFrame()


def calculate_regime_performance(regime_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market performance during each regime period."""
    results = []
    
    # Add end timestamp for each regime (next regime start or last data point)
    regime_df = regime_df.copy()  # Avoid modifying original
    regime_df['end_timestamp'] = regime_df['timestamp'].shift(-1)
    # Convert to same type to avoid dtype mismatch
    last_market_timestamp = pd.Timestamp(market_df.index[-1])
    if hasattr(last_market_timestamp, 'tz_localize') and last_market_timestamp.tz is not None:
        last_market_timestamp = last_market_timestamp.tz_localize(None)
    regime_df.loc[regime_df.index[-1], 'end_timestamp'] = last_market_timestamp
    
    for idx, row in regime_df.iterrows():
        start = row['timestamp']
        end = row['end_timestamp']
        regime = row['regime']
        
        # Get market data for this regime period
        period_data = market_df[(market_df.index >= start) & (market_df.index < end)]
        
        if len(period_data) > 0:
            # Calculate various metrics
            start_price = period_data['close'].iloc[0]
            end_price = period_data['close'].iloc[-1]
            
            results.append({
                'regime': regime,
                'start': start,
                'end': end,
                'duration_hours': len(period_data),
                'return_pct': ((end_price - start_price) / start_price) * 100,
                'volatility': period_data['close'].pct_change().std() * np.sqrt(252 * 6.5),  # Annualized for 1h bars
                'high_low_range': ((period_data['high'].max() - period_data['low'].min()) / start_price) * 100,
                'avg_volume': period_data['volume'].mean(),
                'trend_strength': (end_price - start_price) / period_data['close'].std() if period_data['close'].std() > 0 else 0
            })
    
    return pd.DataFrame(results)


def analyze_optimal_parameters(performance_df: pd.DataFrame) -> Dict[str, Dict]:
    """Determine optimal strategy parameters for each regime."""
    
    # Group by regime
    regime_stats = performance_df.groupby('regime').agg({
        'return_pct': ['mean', 'std', 'count'],
        'volatility': 'mean',
        'duration_hours': 'mean',
        'trend_strength': 'mean',
        'high_low_range': 'mean'
    }).round(3)
    
    # Determine optimal parameters based on regime characteristics
    optimal_params = {}
    
    for regime in regime_stats.index:
        stats = regime_stats.loc[regime]
        
        # Extract key metrics
        avg_return = stats[('return_pct', 'mean')]
        return_std = stats[('return_pct', 'std')]
        avg_volatility = stats[('volatility', 'mean')]
        avg_duration = stats[('duration_hours', 'mean')]
        trend_strength = stats[('trend_strength', 'mean')]
        
        # Determine optimal parameters based on regime
        if regime == 'weak_momentum':
            # Weak momentum - use tighter stops, shorter timeframes
            optimal_params[regime] = {
                'recommended_strategy': 'mean_reversion',
                'parameters': {
                    'entry_threshold': 0.5,  # Enter on smaller moves
                    'exit_bars': max(3, int(avg_duration * 0.3)),  # Exit quickly
                    'stop_loss': 0.5,  # Tight stops
                    'position_size': 0.7,  # Smaller positions due to weak trends
                },
                'reasoning': f"Weak momentum with {avg_return:.2f}% avg return suggests mean reversion. "
                           f"Short duration ({avg_duration:.0f}h) requires quick exits."
            }
            
        elif regime == 'no_momentum':
            # No momentum - avoid trading or use range strategies
            optimal_params[regime] = {
                'recommended_strategy': 'range_bound',
                'parameters': {
                    'entry_threshold': 1.0,  # Only enter on larger moves
                    'exit_bars': max(5, int(avg_duration * 0.5)),
                    'stop_loss': 0.3,  # Very tight stops
                    'position_size': 0.5,  # Conservative sizing
                },
                'reasoning': f"No momentum with {avg_volatility:.1f}% volatility. "
                           f"Best to trade ranges or stay out."
            }
            
        elif regime == 'strong_momentum':
            # Strong momentum - ride the trend
            optimal_params[regime] = {
                'recommended_strategy': 'trend_following',
                'parameters': {
                    'entry_threshold': 0.3,  # Enter early on momentum
                    'exit_bars': max(10, int(avg_duration * 0.7)),  # Hold longer
                    'stop_loss': 1.5,  # Wider stops to avoid whipsaws
                    'position_size': 1.0,  # Full position on strong trends
                    'trailing_stop': True,
                },
                'reasoning': f"Strong momentum with trend strength {trend_strength:.2f}. "
                           f"Let winners run with trailing stops."
            }
        
        # Add performance metrics
        optimal_params[regime]['performance_metrics'] = {
            'avg_return': avg_return,
            'return_volatility': return_std,
            'regime_volatility': avg_volatility,
            'avg_duration_hours': avg_duration,
            'occurrences': int(stats[('return_pct', 'count')])
        }
    
    return optimal_params


def generate_report(regime_df: pd.DataFrame, performance_df: pd.DataFrame, 
                   optimal_params: Dict[str, Dict], metadata: Dict) -> None:
    """Generate a comprehensive regime analysis report."""
    
    print("=" * 80)
    print("REGIME ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Summary statistics
    total_changes = len(regime_df)
    total_bars = metadata.get('total_bars', metadata['signal_statistics'].get('total_positions', 2000))
    regime_breakdown = metadata['signal_statistics']['regime_breakdown']
    
    print(f"Analysis Period: {regime_df['timestamp'].min()} to {regime_df['timestamp'].max()}")
    print(f"Total Bars Analyzed: {total_bars}")
    print(f"Total Regime Changes: {total_changes}")
    
    # Calculate actual average regime duration
    if len(performance_df) > 0:
        avg_duration = performance_df['duration_hours'].mean()
        print(f"Average Regime Duration: {avg_duration:.1f} hours")
    else:
        print(f"Average Regime Duration: {total_bars / total_changes:.1f} bars")
        
    # Check if classifier is too noisy
    if total_changes > total_bars * 0.5:
        print("\n⚠️  WARNING: Classifier is producing too many regime changes!")
        print(f"   {total_changes} changes in {total_bars} bars = regime change every {total_bars/total_changes:.1f} bars")
        print("   Consider adjusting classifier parameters to reduce noise:")
        print("   - Increase momentum_threshold (current: 0.02)")
        print("   - Add regime persistence requirement (min bars in regime)")
        print("   - Use smoothed indicators (EMA instead of raw values)")
        print("   - Add confidence threshold for regime changes")
    print()
    
    print("Regime Distribution:")
    for regime, count in regime_breakdown.items():
        pct = (count / total_changes) * 100
        print(f"  {regime}: {count} occurrences ({pct:.1f}%)")
    print()
    
    # Performance by regime
    print("-" * 80)
    print("PERFORMANCE BY REGIME")
    print("-" * 80)
    
    regime_summary = performance_df.groupby('regime').agg({
        'return_pct': ['mean', 'std', 'min', 'max'],
        'volatility': 'mean',
        'duration_hours': ['mean', 'min', 'max']
    }).round(2)
    
    print(regime_summary)
    print()
    
    # Optimal parameters
    print("-" * 80)
    print("OPTIMAL PARAMETERS BY REGIME")
    print("-" * 80)
    
    for regime, params in optimal_params.items():
        print(f"\n{regime.upper()}:")
        print(f"  Recommended Strategy: {params['recommended_strategy']}")
        print(f"  Reasoning: {params['reasoning']}")
        print("  Parameters:")
        for param, value in params['parameters'].items():
            print(f"    - {param}: {value}")
        print("  Performance Metrics:")
        for metric, value in params['performance_metrics'].items():
            print(f"    - {metric}: {value:.2f}")
    
    print()
    print("-" * 80)
    print("TRADING RECOMMENDATIONS")
    print("-" * 80)
    print()
    print("1. Implement regime-aware position sizing:")
    print("   - Strong momentum: Full position size (1.0x)")
    print("   - Weak momentum: Reduced size (0.7x)")
    print("   - No momentum: Conservative size (0.5x)")
    print()
    print("2. Adjust stop losses based on regime volatility:")
    print("   - Use tighter stops in low momentum regimes")
    print("   - Wider stops in trending regimes to avoid whipsaws")
    print()
    print("3. Consider different strategies per regime:")
    print("   - Trend following in strong momentum")
    print("   - Mean reversion in weak momentum")
    print("   - Range trading or sitting out in no momentum")
    print()


def main():
    """Main analysis function."""
    # Find the most recent classifier output
    classifier_files = list(Path("./workspaces").rglob("*classifier*.json"))
    if not classifier_files:
        print("No classifier output files found!")
        return
    
    # Use the most recent file
    latest_file = max(classifier_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing regime data from: {latest_file}")
    
    # Load regime data
    regime_df, metadata = load_regime_data(latest_file)
    print(f"Loaded {len(regime_df)} regime changes")
    
    # Load market data
    market_df = load_market_data()
    if market_df.empty:
        print("Could not load market data!")
        return
    
    # Map bar indices to market timestamps
    # The metadata shows how many regime changes were captured, not total bars analyzed
    # We requested 2000 bars but only 708 had regime determinations
    total_bars_requested = 2000  # We know we requested 2000 bars
    total_regime_changes = len(regime_df)
    
    print(f"Total bars requested: {total_bars_requested}")
    print(f"Total regime changes captured: {total_regime_changes}")
    
    # For 1h data with 2000 bars, we need to get the right subset
    if len(market_df) < total_bars_requested:
        print(f"Market data has {len(market_df)} bars, but we requested {total_bars_requested} bars")
        total_bars_requested = len(market_df)
    
    # Get the last 'total_bars_requested' from market data
    market_subset = market_df.iloc[-total_bars_requested:].copy()
    market_subset.reset_index(drop=False, inplace=True)
    market_subset['bar_idx'] = range(len(market_subset))
    
    # Map regime bar indices to actual market timestamps
    regime_df = regime_df.merge(
        market_subset[['bar_idx', market_subset.columns[0]]].rename(columns={market_subset.columns[0]: 'market_timestamp'}),
        on='bar_idx',
        how='left'
    )
    
    # Use market timestamps instead of synthetic timestamps
    regime_df['timestamp'] = regime_df['market_timestamp']
    regime_df = regime_df.dropna(subset=['timestamp'])
    
    print(f"Mapped {len(regime_df)} regime changes to market timestamps")
    
    # Calculate performance metrics
    performance_df = calculate_regime_performance(regime_df, market_df)
    
    print(f"\nPerformance data shape: {performance_df.shape}")
    if performance_df.empty:
        print("No performance data calculated! This might be due to a mismatch between regime timestamps and market data.")
        print(f"Regime date range: {regime_df['timestamp'].min()} to {regime_df['timestamp'].max()}")
        print(f"Market date range: {market_df.index.min()} to {market_df.index.max()}")
        return
    
    # Determine optimal parameters
    optimal_params = analyze_optimal_parameters(performance_df)
    
    # Generate report
    generate_report(regime_df, performance_df, optimal_params, metadata)
    
    # Save results
    output_file = "regime_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'regime_summary': performance_df.groupby('regime').mean().to_dict(),
            'optimal_parameters': optimal_params,
            'metadata': metadata
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()