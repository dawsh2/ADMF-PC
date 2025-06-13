#!/usr/bin/env python3
"""
Analyze results from expansive grid search with multiple strategies and classifiers.
"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
import numpy as np

def load_events(file_path):
    """Load events from JSONL file."""
    events = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events

def analyze_grid_search_results(workspace_path):
    """Analyze the grid search results."""
    events_file = Path(workspace_path) / "events.jsonl"
    events = load_events(events_file)
    
    # Separate events by type
    signals = []
    classifications = []
    bars = []
    
    for event in events:
        event_type = event.get('event_type', '')
        if event_type == 'SIGNAL':
            signals.append(event)
        elif event_type == 'CLASSIFICATION':
            classifications.append(event)
        elif event_type == 'BAR':
            bars.append(event)
    
    print(f"\nTotal events: {len(events):,}")
    print(f"BAR events: {len(bars):,}")
    print(f"SIGNAL events: {len(signals):,}")
    print(f"CLASSIFICATION events: {len(classifications):,}")
    
    # Analyze strategies
    strategy_signals = defaultdict(list)
    for signal in signals:
        strategy_id = signal['payload'].get('strategy_id', 'unknown')
        strategy_signals[strategy_id].append(signal)
    
    print(f"\nStrategies found: {len(strategy_signals)}")
    
    # Analyze classifiers and regimes
    classifier_regimes = defaultdict(lambda: defaultdict(int))
    regime_changes = defaultdict(list)
    
    for classification in classifications:
        classifier_id = classification['payload'].get('classifier_id', 'unknown')
        regime = classification['payload'].get('regime', 'unknown')
        classifier_regimes[classifier_id][regime] += 1
        regime_changes[classifier_id].append({
            'timestamp': classification['timestamp'],
            'regime': regime,
            'confidence': classification['payload'].get('confidence', 0)
        })
    
    print(f"\nClassifiers found: {len(classifier_regimes)}")
    for classifier, regimes in classifier_regimes.items():
        print(f"\n{classifier}:")
        for regime, count in sorted(regimes.items()):
            print(f"  {regime}: {count} classifications")
    
    # Analyze strategy performance by regime
    if signals and classifications:
        analyze_regime_performance(signals, classifications, bars)
    
    # Find best parameter combinations
    find_best_parameters(strategy_signals, classifications)

def analyze_regime_performance(signals, classifications, bars):
    """Analyze strategy performance in different regimes."""
    print("\n" + "="*80)
    print("REGIME-BASED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Build price series from bars
    prices = {}
    for bar in bars:
        symbol = bar['payload']['bar']['symbol']
        timestamp = bar['timestamp']
        close_price = bar['payload']['bar']['close']
        prices[timestamp] = close_price
    
    # Sort prices by timestamp
    price_series = pd.Series(prices).sort_index()
    
    # Build regime timeline for each classifier
    regime_timelines = defaultdict(dict)
    for classification in classifications:
        classifier_id = classification['payload']['classifier_id']
        timestamp = classification['timestamp']
        regime = classification['payload']['regime']
        regime_timelines[classifier_id][timestamp] = regime
    
    # Analyze each strategy's performance in different regimes
    strategy_regime_performance = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for signal in signals:
        strategy_id = signal['payload']['strategy_id']
        signal_time = signal['timestamp']
        signal_dir = signal['payload']['direction']
        
        # Find active regime at signal time for each classifier
        for classifier_id, timeline in regime_timelines.items():
            # Get regime at signal time
            regime_times = sorted(timeline.keys())
            active_regime = None
            for t in regime_times:
                if t <= signal_time:
                    active_regime = timeline[t]
                else:
                    break
            
            if active_regime and signal_dir in ['LONG', 'SHORT']:
                # Calculate forward returns
                try:
                    signal_idx = price_series.index.get_loc(signal_time, method='nearest')
                    if signal_idx + 20 < len(price_series):  # Need 20 bars forward
                        entry_price = price_series.iloc[signal_idx]
                        exit_price = price_series.iloc[signal_idx + 20]  # 20 bar holding period
                        
                        if signal_dir == 'LONG':
                            ret = (exit_price - entry_price) / entry_price
                        else:  # SHORT
                            ret = (entry_price - exit_price) / entry_price
                        
                        strategy_regime_performance[strategy_id][classifier_id][active_regime].append(ret)
                except:
                    pass
    
    # Print performance summary
    for strategy_id in sorted(strategy_regime_performance.keys()):
        # Extract strategy type from ID
        strategy_type = strategy_id.split('_')[1] if '_' in strategy_id else strategy_id
        
        print(f"\n{strategy_type.upper()} Strategy Performance by Regime:")
        print("-" * 60)
        
        for classifier_id in sorted(strategy_regime_performance[strategy_id].keys()):
            classifier_name = classifier_id.split('_')[1] if '_' in classifier_id else classifier_id
            print(f"\n  Classifier: {classifier_name}")
            
            regime_stats = []
            for regime, returns in strategy_regime_performance[strategy_id][classifier_id].items():
                if returns:
                    avg_ret = np.mean(returns) * 100
                    win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252/20)  # Annualized
                    regime_stats.append({
                        'regime': regime,
                        'trades': len(returns),
                        'avg_return': avg_ret,
                        'win_rate': win_rate,
                        'sharpe': sharpe
                    })
            
            # Sort by average return
            regime_stats.sort(key=lambda x: x['avg_return'], reverse=True)
            
            for stats in regime_stats:
                print(f"    {stats['regime']:15} | Trades: {stats['trades']:4} | "
                      f"Avg Ret: {stats['avg_return']:6.2f}% | "
                      f"Win Rate: {stats['win_rate']:5.1f}% | "
                      f"Sharpe: {stats['sharpe']:6.2f}")

def find_best_parameters(strategy_signals, classifications):
    """Find best parameter combinations for each strategy type."""
    print("\n" + "="*80)
    print("BEST PARAMETER COMBINATIONS")
    print("="*80)
    
    # Group strategies by type
    strategy_types = defaultdict(list)
    for strategy_id, signals in strategy_signals.items():
        # Extract strategy type
        parts = strategy_id.split('_')
        if len(parts) >= 2:
            strategy_type = parts[1]
            strategy_types[strategy_type].append({
                'id': strategy_id,
                'signals': signals,
                'params': '_'.join(parts[2:]) if len(parts) > 2 else 'default'
            })
    
    # Analyze each strategy type
    for strategy_type, strategies in strategy_types.items():
        print(f"\n{strategy_type.upper()} Strategies ({len(strategies)} variants):")
        print("-" * 60)
        
        # Calculate signal quality metrics for each variant
        variant_metrics = []
        for strategy in strategies:
            total_signals = len(strategy['signals'])
            
            # Count signal types
            long_signals = sum(1 for s in strategy['signals'] 
                             if s['payload'].get('direction') == 'LONG')
            short_signals = sum(1 for s in strategy['signals'] 
                              if s['payload'].get('direction') == 'SHORT')
            
            # Calculate signal rate (signals per bar)
            signal_rate = total_signals / 2000 if total_signals > 0 else 0
            
            variant_metrics.append({
                'params': strategy['params'],
                'total_signals': total_signals,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'signal_rate': signal_rate,
                'long_short_ratio': long_signals / (short_signals + 1)
            })
        
        # Sort by total signals
        variant_metrics.sort(key=lambda x: x['total_signals'], reverse=True)
        
        # Show top 5 most active variants
        print("\nMost Active Parameter Combinations:")
        for i, metrics in enumerate(variant_metrics[:5]):
            print(f"  {i+1}. {metrics['params']}")
            print(f"     Signals: {metrics['total_signals']} "
                  f"(Long: {metrics['long_signals']}, Short: {metrics['short_signals']})")
            print(f"     Signal Rate: {metrics['signal_rate']:.2%} | "
                  f"L/S Ratio: {metrics['long_short_ratio']:.2f}")

if __name__ == "__main__":
    # Use the latest workspace
    workspace = "workspaces/strategy_03e31c53"
    analyze_grid_search_results(workspace)