#!/usr/bin/env python3
"""
Aggregated Regime Analysis Script

Analyzes classifier regime data by aggregating short-lived regimes
to get more meaningful performance statistics.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def aggregate_regimes(regime_df: pd.DataFrame, min_duration_hours: int = 4) -> pd.DataFrame:
    """Aggregate short-lived regimes into longer periods."""
    aggregated = []
    
    if len(regime_df) == 0:
        return pd.DataFrame()
    
    # Start with first regime
    current_regime = {
        'regime': regime_df.iloc[0]['regime'],
        'start': regime_df.iloc[0]['timestamp'],
        'end': regime_df.iloc[0]['timestamp'],
        'changes': 1,
        'regimes_seen': [regime_df.iloc[0]['regime']]
    }
    
    for idx in range(1, len(regime_df)):
        row = regime_df.iloc[idx]
        
        # Calculate hours since regime start
        hours_in_regime = (row['timestamp'] - current_regime['start']).total_seconds() / 3600
        
        # If we've been in this aggregated period long enough, start a new one
        if hours_in_regime >= min_duration_hours:
            # Determine dominant regime
            regime_counts = pd.Series(current_regime['regimes_seen']).value_counts()
            current_regime['regime'] = regime_counts.index[0]
            current_regime['end'] = regime_df.iloc[idx-1]['timestamp']
            aggregated.append(current_regime.copy())
            
            # Start new regime
            current_regime = {
                'regime': row['regime'],
                'start': row['timestamp'],
                'end': row['timestamp'],
                'changes': 1,
                'regimes_seen': [row['regime']]
            }
        else:
            # Continue aggregating
            current_regime['regimes_seen'].append(row['regime'])
            current_regime['changes'] += 1
    
    # Add final regime
    if current_regime['changes'] > 0:
        regime_counts = pd.Series(current_regime['regimes_seen']).value_counts()
        current_regime['regime'] = regime_counts.index[0]
        current_regime['end'] = regime_df.iloc[-1]['timestamp']
        aggregated.append(current_regime)
    
    # Convert to DataFrame
    agg_df = pd.DataFrame(aggregated)
    if not agg_df.empty:
        agg_df['duration_hours'] = (agg_df['end'] - agg_df['start']).dt.total_seconds() / 3600
        agg_df = agg_df[['regime', 'start', 'end', 'duration_hours', 'changes']]
    
    return agg_df


def analyze_aggregated_performance(agg_regime_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market performance for aggregated regime periods."""
    results = []
    
    for idx, row in agg_regime_df.iterrows():
        start = row['start']
        end = row['end']
        regime = row['regime']
        
        # Get market data for this regime period
        period_data = market_df[(market_df.index >= start) & (market_df.index <= end)]
        
        if len(period_data) > 1:
            # Calculate various metrics
            start_price = period_data['close'].iloc[0]
            end_price = period_data['close'].iloc[-1]
            
            # Calculate returns
            period_return = ((end_price - start_price) / start_price) * 100
            
            # Calculate volatility (annualized)
            returns = period_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252 * 6.5) if len(returns) > 0 else 0
            
            # Calculate trend metrics
            high_low_range = ((period_data['high'].max() - period_data['low'].min()) / start_price) * 100
            
            # Calculate drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = ((cumulative - running_max) / running_max).min() * 100 if len(cumulative) > 0 else 0
            
            results.append({
                'regime': regime,
                'start': start,
                'end': end,
                'duration_hours': row['duration_hours'],
                'changes_within': row['changes'],
                'return_pct': period_return,
                'volatility': volatility * 100,  # Convert to percentage
                'high_low_range': high_low_range,
                'max_drawdown': drawdown,
                'bars': len(period_data),
                'avg_volume': period_data['volume'].mean()
            })
    
    return pd.DataFrame(results)


def generate_aggregated_report(agg_regime_df: pd.DataFrame, performance_df: pd.DataFrame) -> None:
    """Generate report for aggregated regimes."""
    
    print("\n" + "=" * 80)
    print("AGGREGATED REGIME ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    print(f"Total Aggregated Regimes: {len(agg_regime_df)}")
    print(f"Average Regime Duration: {agg_regime_df['duration_hours'].mean():.1f} hours")
    print(f"Total Period: {agg_regime_df['start'].min()} to {agg_regime_df['end'].max()}")
    print()
    
    # Regime distribution
    regime_dist = agg_regime_df.groupby('regime')['duration_hours'].agg(['count', 'sum', 'mean'])
    regime_dist['pct_time'] = (regime_dist['sum'] / regime_dist['sum'].sum() * 100).round(1)
    
    print("Regime Time Distribution:")
    for regime in regime_dist.index:
        stats = regime_dist.loc[regime]
        print(f"  {regime:20s}: {stats['count']:3.0f} periods, "
              f"{stats['sum']:6.1f} hours total ({stats['pct_time']:4.1f}%), "
              f"avg {stats['mean']:4.1f} hours/period")
    print()
    
    # Performance by regime
    if len(performance_df) > 0:
        print("-" * 80)
        print("PERFORMANCE BY AGGREGATED REGIME")
        print("-" * 80)
        
        perf_summary = performance_df.groupby('regime').agg({
            'return_pct': ['mean', 'std', 'min', 'max', 'count'],
            'volatility': 'mean',
            'max_drawdown': 'mean',
            'duration_hours': 'mean'
        }).round(2)
        
        print(perf_summary)
        print()
        
        # Best and worst periods
        print("-" * 80)
        print("NOTABLE REGIME PERIODS")
        print("-" * 80)
        
        if len(performance_df) >= 3:
            # Best performing periods
            best_periods = performance_df.nlargest(3, 'return_pct')[['regime', 'start', 'end', 'duration_hours', 'return_pct']]
            print("\nBest Performing Periods:")
            for idx, row in best_periods.iterrows():
                print(f"  {row['regime']:20s} {row['start'].strftime('%Y-%m-%d %H:%M')} to "
                      f"{row['end'].strftime('%Y-%m-%d %H:%M')} "
                      f"({row['duration_hours']:.1f}h): {row['return_pct']:+.2f}%")
            
            # Worst performing periods
            worst_periods = performance_df.nsmallest(3, 'return_pct')[['regime', 'start', 'end', 'duration_hours', 'return_pct']]
            print("\nWorst Performing Periods:")
            for idx, row in worst_periods.iterrows():
                print(f"  {row['regime']:20s} {row['start'].strftime('%Y-%m-%d %H:%M')} to "
                      f"{row['end'].strftime('%Y-%m-%d %H:%M')} "
                      f"({row['duration_hours']:.1f}h): {row['return_pct']:+.2f}%")
        
        # Regime-specific insights
        print("\n" + "-" * 80)
        print("REGIME-SPECIFIC INSIGHTS")
        print("-" * 80)
        
        for regime in performance_df['regime'].unique():
            regime_data = performance_df[performance_df['regime'] == regime]
            if len(regime_data) > 0:
                print(f"\n{regime.upper()}:")
                
                avg_return = regime_data['return_pct'].mean()
                win_rate = (regime_data['return_pct'] > 0).mean() * 100
                avg_duration = regime_data['duration_hours'].mean()
                
                print(f"  Average Return: {avg_return:+.2f}%")
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Average Duration: {avg_duration:.1f} hours")
                print(f"  Average Volatility: {regime_data['volatility'].mean():.1f}%")
                
                # Recommendations based on performance
                if avg_return > 0.5:
                    print(f"  ✓ Strong performance - consider trend following strategies")
                elif avg_return < -0.5:
                    print(f"  ⚠ Poor performance - consider defensive strategies or staying flat")
                else:
                    print(f"  → Neutral performance - consider range-bound strategies")


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
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Extract regime changes
    changes = []
    for i, change in enumerate(data['changes']):
        changes.append({
            'bar_idx': change['idx'],
            'timestamp': pd.to_datetime(change['ts']).tz_localize(None),
            'symbol': change['sym'],
            'regime': change['val'],
            'strategy': change['strat']
        })
    
    regime_df = pd.DataFrame(changes)
    print(f"Loaded {len(regime_df)} regime changes")
    
    # Load market data
    from analyze_regimes import load_market_data
    market_df = load_market_data()
    if market_df.empty:
        print("Could not load market data!")
        return
    
    # Map bar indices to market timestamps
    total_bars = data['metadata'].get('total_bars', 2000)
    if len(market_df) < total_bars:
        total_bars = len(market_df)
    
    # Get the last 'total_bars' from market data
    market_subset = market_df.iloc[-total_bars:].copy()
    market_subset.reset_index(drop=False, inplace=True)
    market_subset['bar_idx'] = range(len(market_subset))
    
    # Map regime bar indices to actual market timestamps
    regime_df = regime_df.merge(
        market_subset[['bar_idx', market_subset.columns[0]]].rename(columns={market_subset.columns[0]: 'market_timestamp'}),
        on='bar_idx',
        how='left'
    )
    
    # Use market timestamps
    regime_df['timestamp'] = regime_df['market_timestamp']
    regime_df = regime_df.dropna(subset=['timestamp'])
    
    print(f"Mapped {len(regime_df)} regime changes to market timestamps")
    
    # Aggregate regimes (minimum 4 hours per regime)
    agg_regime_df = aggregate_regimes(regime_df, min_duration_hours=4)
    print(f"Aggregated to {len(agg_regime_df)} regime periods")
    
    # Calculate performance for aggregated regimes
    performance_df = analyze_aggregated_performance(agg_regime_df, market_df)
    
    # Generate report
    generate_aggregated_report(agg_regime_df, performance_df)
    
    # Save results
    output_file = "aggregated_regime_analysis_results.json"
    results = {
        'aggregated_regimes': agg_regime_df.to_dict('records'),
        'performance': performance_df.to_dict('records') if len(performance_df) > 0 else [],
        'summary': {
            'total_regimes': len(agg_regime_df),
            'avg_duration_hours': agg_regime_df['duration_hours'].mean() if len(agg_regime_df) > 0 else 0,
            'regime_distribution': agg_regime_df['regime'].value_counts().to_dict() if len(agg_regime_df) > 0 else {}
        }
    }
    
    # Convert timestamps to strings for JSON serialization
    for regime in results['aggregated_regimes']:
        regime['start'] = str(regime['start'])
        regime['end'] = str(regime['end'])
    for perf in results['performance']:
        perf['start'] = str(perf['start'])
        perf['end'] = str(perf['end'])
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()