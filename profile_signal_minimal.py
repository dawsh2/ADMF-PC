#!/usr/bin/env python3
"""
Minimal profiling test to identify bottlenecks in signal generation.
"""

import time
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timing data
timing_data = defaultdict(list)

def profile_feature_computation():
    """Profile feature computation performance."""
    from src.strategy.components.features.hub import FeatureHub, compute_feature, FEATURE_REGISTRY
    
    print("\n=== PROFILING FEATURE COMPUTATION ===")
    
    # Create test data
    n_bars = 100
    test_data = pd.DataFrame({
        'open': np.random.randn(n_bars).cumsum() + 100,
        'high': np.random.randn(n_bars).cumsum() + 101,
        'low': np.random.randn(n_bars).cumsum() + 99,
        'close': np.random.randn(n_bars).cumsum() + 100,
        'volume': np.random.randint(1000000, 5000000, n_bars)
    })
    
    # Define typical features used by strategies
    feature_configs = {
        # SMA features (used by MA crossover strategies)
        'sma_5': {'feature': 'sma', 'period': 5},
        'sma_10': {'feature': 'sma', 'period': 10},
        'sma_20': {'feature': 'sma', 'period': 20},
        'sma_50': {'feature': 'sma', 'period': 50},
        'sma_100': {'feature': 'sma', 'period': 100},
        'sma_200': {'feature': 'sma', 'period': 200},
        
        # RSI features (used by momentum strategies)
        'rsi_7': {'feature': 'rsi', 'period': 7},
        'rsi_14': {'feature': 'rsi', 'period': 14},
        'rsi_21': {'feature': 'rsi', 'period': 21},
        
        # Bollinger bands (used by mean reversion)
        'bollinger_20': {'feature': 'bollinger_bands', 'period': 20, 'std_dev': 2.0},
        
        # ATR (used by breakout strategies)
        'atr_14': {'feature': 'atr', 'period': 14},
        
        # Volume features
        'volume_20_volume_ma': {'feature': 'volume_sma', 'period': 20},
        'volume_50_volume_ma': {'feature': 'volume_sma', 'period': 50},
        
        # High/Low features (for breakout)
        'high_20': {'feature': 'high', 'lookback_period': 20},
        'high_50': {'feature': 'high', 'lookback_period': 50},
        'low_20': {'feature': 'low', 'lookback_period': 20},
        'low_50': {'feature': 'low', 'lookback_period': 50},
    }
    
    # Time individual feature computation
    print(f"\nComputing {len(feature_configs)} features on {n_bars} bars:")
    
    feature_times = {}
    for feature_name, config in feature_configs.items():
        feature_type = config.get('feature')
        params = {k: v for k, v in config.items() if k != 'feature'}
        
        start = time.perf_counter()
        try:
            result = compute_feature(feature_type, test_data, **params)
            duration = time.perf_counter() - start
            feature_times[feature_name] = duration * 1000  # Convert to ms
            print(f"  {feature_name}: {duration*1000:.2f}ms")
        except Exception as e:
            print(f"  {feature_name}: ERROR - {e}")
    
    # Test incremental update performance
    print("\n=== TESTING INCREMENTAL UPDATES ===")
    
    hub = FeatureHub(['SPY'])
    hub.configure_features(feature_configs)
    
    # Simulate streaming bars
    update_times = []
    for i in range(20):
        bar = {
            'open': 100 + np.random.randn(),
            'high': 101 + np.random.randn(),
            'low': 99 + np.random.randn(),
            'close': 100 + np.random.randn(),
            'volume': np.random.randint(1000000, 5000000)
        }
        
        start = time.perf_counter()
        hub.update_bar('SPY', bar)
        duration = time.perf_counter() - start
        update_times.append(duration * 1000)
        
        if i % 5 == 0:
            features = hub.get_features('SPY')
            print(f"  Bar {i+1}: update={duration*1000:.2f}ms, features available={len(features)}")
    
    print(f"\nIncremental update stats:")
    print(f"  Mean: {np.mean(update_times):.2f}ms")
    print(f"  Median: {np.median(update_times):.2f}ms")
    print(f"  Max: {max(update_times):.2f}ms")
    
    return feature_times, update_times


def profile_strategy_execution():
    """Profile strategy execution performance."""
    print("\n=== PROFILING STRATEGY EXECUTION ===")
    
    # Import strategy functions
    from src.strategy.strategies.ma_crossover import ma_crossover_strategy
    from src.strategy.strategies.rsi_strategy import rsi_strategy
    from src.strategy.strategies.momentum_strategy import momentum_strategy
    from src.strategy.strategies.mean_reversion_simple import mean_reversion_simple_strategy
    from src.strategy.strategies.breakout_strategy import breakout_strategy
    
    # Create test features and bar
    test_features = {
        'sma_5': 100.5,
        'sma_10': 100.3,
        'sma_20': 100.1,
        'sma_50': 99.9,
        'sma_100': 99.7,
        'sma_200': 99.5,
        'rsi_7': 55,
        'rsi_14': 52,
        'rsi_21': 50,
        'bollinger_20_upper': 102,
        'bollinger_20_lower': 98,
        'bollinger_20_middle': 100,
        'atr_14': 1.5,
        'volume_20_volume_ma': 2000000,
        'volume_50_volume_ma': 2500000,
        'high_20': 101.5,
        'high_50': 102,
        'low_20': 98.5,
        'low_50': 98,
        'bar_count': 100
    }
    
    test_bar = {
        'open': 100.1,
        'high': 100.5,
        'low': 99.8,
        'close': 100.3,
        'volume': 2100000
    }
    
    # Test different strategy types
    strategies = [
        ('ma_crossover', ma_crossover_strategy, {'fast_period': 10, 'slow_period': 20}),
        ('rsi', rsi_strategy, {'rsi_period': 14, 'oversold': 30, 'overbought': 70}),
        ('momentum', momentum_strategy, {'sma_period': 20, 'rsi_period': 14, 'rsi_threshold': 50}),
        ('mean_reversion', mean_reversion_simple_strategy, {'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14}),
        ('breakout', breakout_strategy, {'lookback_period': 20, 'volume_multiplier': 1.5, 'atr_multiplier': 2.0})
    ]
    
    # Time individual strategies
    print("\nTiming individual strategy executions (1000 calls each):")
    strategy_times = {}
    
    for name, func, params in strategies:
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            result = func(test_features, test_bar, params)
            duration = time.perf_counter() - start
            times.append(duration * 1000)
        
        strategy_times[name] = times
        print(f"  {name}: mean={np.mean(times):.3f}ms, median={np.median(times):.3f}ms, max={max(times):.3f}ms")
    
    # Test with missing features
    print("\nTesting with minimal features:")
    minimal_features = {
        'sma_10': 100.3,
        'sma_20': 100.1,
        'rsi_14': 52,
        'bar_count': 100
    }
    
    for name, func, params in strategies[:2]:  # Just test first two
        try:
            start = time.perf_counter()
            result = func(minimal_features, test_bar, params)
            duration = time.perf_counter() - start
            print(f"  {name}: {duration*1000:.3f}ms (result: {result})")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    return strategy_times


def profile_component_state():
    """Profile ComponentState readiness checks and execution."""
    print("\n=== PROFILING COMPONENT STATE ===")
    
    from src.strategy.state import ComponentState
    from src.strategy.strategies.ma_crossover import ma_crossover_strategy
    
    # Create component state
    state = ComponentState(['SPY'])
    
    # Add multiple strategies with different parameters
    print("\nAdding 1000 MA crossover strategies with different parameters...")
    
    strategy_count = 0
    for fast in [3, 5, 10, 15, 20]:
        for slow in [10, 20, 30, 50, 100, 200]:
            if fast < slow:
                for threshold in [0.001, 0.002, 0.003]:
                    strategy_id = f"SPY_ma_crossover_{fast}_{slow}_{threshold}"
                    state.add_component(
                        component_id=strategy_id,
                        component_func=ma_crossover_strategy,
                        component_type="strategy",
                        parameters={
                            'fast_period': fast,
                            'slow_period': slow,
                            'threshold': threshold
                        }
                    )
                    strategy_count += 1
    
    print(f"Added {strategy_count} strategies")
    
    # Create test features
    test_features = {f'sma_{i}': 100 + np.random.randn() for i in [3, 5, 10, 15, 20, 30, 50, 100, 200]}
    test_features['bar_count'] = 210  # Ensure all strategies are ready
    
    test_bar = {
        'open': 100.1,
        'high': 100.5,
        'low': 99.8,
        'close': 100.3,
        'volume': 2100000
    }
    
    # Time readiness checks
    print("\nTiming readiness checks:")
    readiness_times = []
    
    for _ in range(10):
        start = time.perf_counter()
        
        components_snapshot = list(state._components.items())
        ready_count = 0
        
        for component_id, component_info in components_snapshot:
            if state._is_component_ready(component_id, component_info, test_features, 210):
                ready_count += 1
        
        duration = time.perf_counter() - start
        readiness_times.append(duration * 1000)
        print(f"  Check: {duration*1000:.2f}ms ({ready_count} ready)")
    
    print(f"\nReadiness check stats:")
    print(f"  Mean: {np.mean(readiness_times):.2f}ms")
    print(f"  For {strategy_count} strategies: {np.mean(readiness_times)/strategy_count:.3f}ms per strategy")
    
    # Time full execution
    print("\nTiming full execution cycle:")
    execution_times = []
    
    for i in range(5):
        start = time.perf_counter()
        state._execute_components_individually('SPY', test_features, test_bar, datetime.now())
        duration = time.perf_counter() - start
        execution_times.append(duration * 1000)
        print(f"  Execution {i+1}: {duration*1000:.2f}ms")
    
    print(f"\nExecution stats:")
    print(f"  Mean: {np.mean(execution_times):.2f}ms")
    print(f"  Per strategy: {np.mean(execution_times)/strategy_count:.3f}ms")
    
    return readiness_times, execution_times


def main():
    """Run all profiling tests."""
    print("="*80)
    print("SIGNAL GENERATION PERFORMANCE PROFILING")
    print("="*80)
    
    # Profile features
    feature_times, update_times = profile_feature_computation()
    
    # Profile strategies
    strategy_times = profile_strategy_execution()
    
    # Profile component state
    readiness_times, execution_times = profile_component_state()
    
    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print("\nFeature Computation (per feature):")
    sorted_features = sorted(feature_times.items(), key=lambda x: x[1], reverse=True)
    for feat, time_ms in sorted_features[:5]:
        print(f"  {feat}: {time_ms:.2f}ms")
    
    print("\nStrategy Execution (per call):")
    for strat, times in strategy_times.items():
        print(f"  {strat}: {np.mean(times):.3f}ms")
    
    print("\nComponent State (1000 strategies):")
    print(f"  Readiness check: {np.mean(readiness_times):.2f}ms total, {np.mean(readiness_times)/1000:.3f}ms per strategy")
    print(f"  Full execution: {np.mean(execution_times):.2f}ms total, {np.mean(execution_times)/1000:.3f}ms per strategy")
    
    # Estimate total time for 1000 strategies over 1000 bars
    feature_time_per_bar = np.mean(update_times)
    readiness_time_per_bar = np.mean(readiness_times)
    execution_time_per_bar = np.mean(execution_times)
    total_per_bar = feature_time_per_bar + readiness_time_per_bar + execution_time_per_bar
    
    print(f"\nEstimated performance for 1000 strategies:")
    print(f"  Per bar: {total_per_bar:.2f}ms")
    print(f"  1000 bars: {total_per_bar * 1000 / 1000:.2f}s")
    print(f"  10000 bars: {total_per_bar * 10000 / 1000:.2f}s")


if __name__ == "__main__":
    main()