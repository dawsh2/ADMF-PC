#!/usr/bin/env python3
"""
Analyze regime transitions and their impact on ensemble performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_data_and_signals(workspace_path: str, start_idx: int, end_idx: int):
    """Load all necessary data and signals."""
    # Load SPY data
    spy = pd.read_parquet('/Users/daws/ADMF-PC/data/SPY_1m.parquet')
    spy_subset = spy.loc[start_idx:end_idx].copy()
    
    # Load classifier
    classifier_df = pd.read_parquet(f"{workspace_path}/traces/SPY_1m/classifiers/unknown/SPY_vol_mom_classifier.parquet")
    classifier_filtered = classifier_df[(classifier_df['idx'] >= start_idx) & (classifier_df['idx'] <= end_idx)]
    
    # Load ensemble signals
    signals = {}
    for ensemble in ['default', 'custom']:
        df = pd.read_parquet(f"{workspace_path}/traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_{ensemble}.parquet")
        signals[ensemble] = df[(df['idx'] >= start_idx) & (df['idx'] <= end_idx)]
    
    return spy_subset, classifier_filtered, signals

def analyze_regime_transitions(classifier_data: pd.DataFrame, spy_data: pd.DataFrame) -> pd.DataFrame:
    """Analyze regime transitions and market conditions."""
    # Create regime timeline
    regime_changes = []
    
    for i, row in classifier_data.iterrows():
        idx = row['idx']
        regime = row['val']
        timestamp = pd.Timestamp(row['ts'])
        
        # Get market data at transition
        if idx in spy_data.index:
            price = spy_data.loc[idx, 'close']
            volume = spy_data.loc[idx, 'volume']
            
            # Calculate recent volatility
            if idx >= 20:
                recent_returns = spy_data.loc[idx-20:idx, 'close'].pct_change()
                volatility = recent_returns.std() * np.sqrt(390 * 252)
            else:
                volatility = np.nan
            
            regime_changes.append({
                'idx': idx,
                'timestamp': timestamp,
                'regime': regime,
                'price': price,
                'volume': volume,
                'volatility': volatility
            })
    
    return pd.DataFrame(regime_changes)

def calculate_regime_performance(spy_data: pd.DataFrame, regime_transitions: pd.DataFrame) -> Dict:
    """Calculate market performance during each regime period."""
    regime_performance = []
    
    for i in range(len(regime_transitions) - 1):
        start_idx = regime_transitions.iloc[i]['idx']
        end_idx = regime_transitions.iloc[i + 1]['idx']
        regime = regime_transitions.iloc[i]['regime']
        
        # Get price data for this regime period
        regime_data = spy_data.loc[start_idx:end_idx]
        
        if len(regime_data) > 0:
            # Calculate metrics
            period_return = (regime_data['close'].iloc[-1] / regime_data['close'].iloc[0]) - 1
            period_volatility = regime_data['close'].pct_change().std() * np.sqrt(390 * 252)
            avg_volume = regime_data['volume'].mean()
            duration_bars = len(regime_data)
            
            regime_performance.append({
                'regime': regime,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'duration_bars': duration_bars,
                'return': period_return,
                'volatility': period_volatility,
                'avg_volume': avg_volume
            })
    
    return pd.DataFrame(regime_performance)

def analyze_signal_clustering(signals_df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Analyze how signals cluster together."""
    if len(signals_df) == 0:
        return pd.DataFrame()
    
    # Sort by index
    signals_sorted = signals_df.sort_values('idx')
    
    # Calculate time between signals
    signals_sorted['bars_since_last'] = signals_sorted['idx'].diff()
    
    # Rolling statistics
    clustering_stats = []
    
    for i in range(window, len(signals_sorted), window//2):
        window_signals = signals_sorted.iloc[max(0, i-window):i]
        
        clustering_stats.append({
            'idx': signals_sorted.iloc[i]['idx'],
            'avg_gap': window_signals['bars_since_last'].mean(),
            'signal_density': len(window_signals) / window,
            'signal_changes': len(window_signals)
        })
    
    return pd.DataFrame(clustering_stats)

def main():
    workspace_path = '/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_9c2c22c9'
    
    # Define analysis period (last 22k bars)
    start_idx = 80236
    end_idx = 102235
    
    print("Loading data...")
    spy_data, classifier_data, signals = load_data_and_signals(workspace_path, start_idx, end_idx)
    
    # Analyze regime transitions
    print("\n=== Regime Transition Analysis ===")
    regime_transitions = analyze_regime_transitions(classifier_data, spy_data)
    print(f"Total regime changes: {len(regime_transitions)}")
    
    # Show transition patterns
    print("\nRegime Transition Matrix:")
    if len(regime_transitions) > 1:
        transitions = []
        for i in range(len(regime_transitions) - 1):
            from_regime = regime_transitions.iloc[i]['regime']
            to_regime = regime_transitions.iloc[i + 1]['regime']
            transitions.append((from_regime, to_regime))
        
        transition_counts = pd.Series(transitions).value_counts()
        print(transition_counts)
    
    # Calculate regime performance
    print("\n=== Market Performance by Regime ===")
    regime_performance = calculate_regime_performance(spy_data, regime_transitions)
    
    for regime in regime_performance['regime'].unique():
        regime_data = regime_performance[regime_performance['regime'] == regime]
        print(f"\n{regime}:")
        print(f"  Occurrences: {len(regime_data)}")
        print(f"  Avg Duration: {regime_data['duration_bars'].mean():.0f} bars")
        print(f"  Avg Return: {regime_data['return'].mean():.3%}")
        print(f"  Avg Volatility: {regime_data['volatility'].mean():.1%}")
        print(f"  Total Bars: {regime_data['duration_bars'].sum()}")
    
    # Analyze signal clustering
    print("\n=== Signal Clustering Analysis ===")
    for ensemble_type in ['default', 'custom']:
        print(f"\n{ensemble_type.upper()} Ensemble:")
        
        clustering = analyze_signal_clustering(signals[ensemble_type], window=200)
        if len(clustering) > 0:
            print(f"  Average gap between signals: {clustering['avg_gap'].mean():.1f} bars")
            print(f"  Signal density (signals per bar): {clustering['signal_density'].mean():.3f}")
            
            # High activity periods
            high_activity = clustering[clustering['signal_density'] > clustering['signal_density'].quantile(0.9)]
            print(f"  High activity periods: {len(high_activity)}")
    
    # Analyze signal behavior around regime changes
    print("\n=== Signal Behavior Around Regime Changes ===")
    
    for ensemble_type in ['default', 'custom']:
        print(f"\n{ensemble_type.upper()} Ensemble:")
        
        signals_df = signals[ensemble_type]
        
        # For each regime change, count signals before and after
        window = 50  # bars before/after regime change
        
        pre_change_signals = []
        post_change_signals = []
        
        for _, transition in regime_transitions.iterrows():
            transition_idx = transition['idx']
            
            # Count signals in window before change
            pre_signals = signals_df[(signals_df['idx'] >= transition_idx - window) & 
                                    (signals_df['idx'] < transition_idx)]
            pre_change_signals.append(len(pre_signals))
            
            # Count signals in window after change
            post_signals = signals_df[(signals_df['idx'] > transition_idx) & 
                                     (signals_df['idx'] <= transition_idx + window)]
            post_change_signals.append(len(post_signals))
        
        if pre_change_signals:
            print(f"  Avg signals {window} bars before regime change: {np.mean(pre_change_signals):.1f}")
            print(f"  Avg signals {window} bars after regime change: {np.mean(post_change_signals):.1f}")
            print(f"  Signal increase after regime change: {(np.mean(post_change_signals) / max(np.mean(pre_change_signals), 1) - 1) * 100:.1f}%")
    
    # Market regime statistics
    print("\n=== Market Statistics by Regime ===")
    
    # Create full regime timeline
    regime_timeline = pd.Series('neutral', index=spy_data.index)
    
    for i, row in classifier_data.iterrows():
        idx = row['idx']
        regime = row['val']
        # Forward fill from this point
        regime_timeline.loc[idx:] = regime
    
    # Calculate returns by regime
    spy_data['regime'] = regime_timeline
    spy_data['return'] = spy_data['close'].pct_change()
    
    regime_stats = spy_data.groupby('regime').agg({
        'return': ['mean', 'std', 'count'],
        'volume': 'mean',
        'close': lambda x: (x.iloc[-1] / x.iloc[0] - 1) if len(x) > 0 else 0
    })
    
    print("\nDetailed Regime Statistics:")
    print(regime_stats)
    
    # Drawdown analysis by regime
    print("\n=== Drawdown Analysis by Regime ===")
    
    for regime in spy_data['regime'].unique():
        regime_data = spy_data[spy_data['regime'] == regime]['close']
        if len(regime_data) > 0:
            cumulative = (1 + regime_data.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            print(f"{regime}: Max Drawdown = {max_dd:.2%}")

if __name__ == "__main__":
    main()