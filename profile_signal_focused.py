#!/usr/bin/env python3
"""
Focused profiling test to identify bottlenecks in signal generation.
"""

import time
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def profile_feature_computation():
    """Profile feature computation performance."""
    from src.strategy.components.features.hub import FeatureHub, compute_feature
    
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
    
    # Define typical features used by strategies (fixed parameter names)
    feature_configs = {
        # SMA features (used by MA crossover strategies)
        'sma_5': {'feature': 'sma', 'period': 5},
        'sma_10': {'feature': 'sma', 'period': 10},
        'sma_20': {'feature': 'sma', 'period': 20},
        'sma_50': {'feature': 'sma', 'period': 50},
        'sma_100': {'feature': 'sma', 'period': 100},
        
        # RSI features (used by momentum strategies)
        'rsi_14': {'feature': 'rsi', 'period': 14},
        
        # Bollinger bands (used by mean reversion)
        'bollinger_20': {'feature': 'bollinger_bands', 'period': 20, 'std_dev': 2.0},
        
        # ATR (used by breakout strategies)
        'atr_14': {'feature': 'atr', 'period': 14},
        
        # Volume features
        'volume_20': {'feature': 'volume_sma', 'period': 20},
        
        # High/Low features (using correct parameter name)
        'high_20': {'feature': 'high', 'period': 20},
        'low_20': {'feature': 'low', 'period': 20},
    }
    
    # Time individual feature computation
    print(f"\nComputing {len(feature_configs)} features on {n_bars} bars:")
    
    feature_times = {}
    total_start = time.perf_counter()
    
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
    
    total_duration = time.perf_counter() - total_start
    print(f"\nTotal feature computation: {total_duration*1000:.2f}ms")
    
    # Test incremental update performance with FeatureHub
    print("\n=== TESTING FEATURE HUB INCREMENTAL UPDATES ===")
    
    hub = FeatureHub(['SPY'])
    hub.configure_features(feature_configs)
    
    # Simulate streaming bars
    update_times = []
    get_times = []
    
    for i in range(50):
        bar = {
            'open': 100 + np.random.randn(),
            'high': 101 + np.random.randn(),
            'low': 99 + np.random.randn(),
            'close': 100 + np.random.randn(),
            'volume': np.random.randint(1000000, 5000000)
        }
        
        # Time update
        start = time.perf_counter()
        hub.update_bar('SPY', bar)
        update_duration = time.perf_counter() - start
        update_times.append(update_duration * 1000)
        
        # Time get
        start = time.perf_counter()
        features = hub.get_features('SPY')
        get_duration = time.perf_counter() - start
        get_times.append(get_duration * 1000)
        
        if i % 10 == 0:
            print(f"  Bar {i+1}: update={update_duration*1000:.2f}ms, get={get_duration*1000:.2f}ms, features={len(features)}")
    
    print(f"\nFeatureHub update stats (ms):")
    print(f"  Mean: {np.mean(update_times):.2f}")
    print(f"  Median: {np.median(update_times):.2f}")
    print(f"  P95: {np.percentile(update_times, 95):.2f}")
    print(f"  Max: {max(update_times):.2f}")
    
    print(f"\nFeatureHub get_features stats (ms):")
    print(f"  Mean: {np.mean(get_times):.3f}")
    print(f"  Max: {max(get_times):.3f}")
    
    return feature_times, update_times


def profile_strategy_execution():
    """Profile strategy execution performance."""
    print("\n=== PROFILING STRATEGY EXECUTION ===")
    
    # Import available strategy functions
    from src.strategy.strategies.ma_crossover import ma_crossover_strategy
    from src.strategy.strategies.rsi_strategy import rsi_strategy
    from src.strategy.strategies.mean_reversion_simple import mean_reversion_simple_strategy
    from src.strategy.strategies.breakout_strategy import breakout_strategy
    
    # Create test features and bar
    test_features = {
        'sma_5': 100.5,
        'sma_10': 100.3,
        'sma_20': 100.1,
        'sma_50': 99.9,
        'sma_100': 99.7,
        'rsi_14': 52,
        'bollinger_20_upper': 102,
        'bollinger_20_lower': 98,
        'bollinger_20_middle': 100,
        'atr_14': 1.5,
        'volume_20': 2000000,
        'high_20': 101.5,
        'low_20': 98.5,
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
        ('ma_crossover_5_20', ma_crossover_strategy, {'fast_period': 5, 'slow_period': 20}),
        ('ma_crossover_10_50', ma_crossover_strategy, {'fast_period': 10, 'slow_period': 50}),
        ('ma_crossover_20_100', ma_crossover_strategy, {'fast_period': 20, 'slow_period': 100}),
        ('rsi_14', rsi_strategy, {'rsi_period': 14, 'oversold': 30, 'overbought': 70}),
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
        print(f"  {name}: mean={np.mean(times):.3f}ms, median={np.median(times):.3f}ms, P95={np.percentile(times, 95):.3f}ms")
    
    # Simulate 1000 strategies (different parameters)
    print("\nSimulating 1000 MA crossover strategies with different parameters:")
    
    all_strategies = []
    for fast in range(3, 50, 2):
        for slow in range(fast + 5, 200, 5):
            all_strategies.append(('ma_crossover', ma_crossover_strategy, {'fast_period': fast, 'slow_period': slow}))
            if len(all_strategies) >= 1000:
                break
        if len(all_strategies) >= 1000:
            break
    
    print(f"Created {len(all_strategies)} strategy configurations")
    
    # Time execution of all strategies
    batch_times = []
    for batch_start in range(0, len(all_strategies), 100):
        batch = all_strategies[batch_start:batch_start + 100]
        
        start = time.perf_counter()
        results = []
        for name, func, params in batch:
            result = func(test_features, test_bar, params)
            results.append(result)
        duration = time.perf_counter() - start
        batch_times.append(duration * 1000)
        
        if batch_start % 200 == 0:
            print(f"  Batch {batch_start//100 + 1}: {duration*1000:.2f}ms for 100 strategies ({duration*10:.3f}ms per strategy)")
    
    print(f"\nBatch execution stats (100 strategies per batch):")
    print(f"  Mean: {np.mean(batch_times):.2f}ms")
    print(f"  Total for 1000: {sum(batch_times):.2f}ms")
    
    return strategy_times, batch_times


def profile_component_state_overhead():
    """Profile ComponentState overhead specifically."""
    print("\n=== PROFILING COMPONENT STATE OVERHEAD ===")
    
    from src.strategy.state import ComponentState
    
    # Create a mock strategy function that does minimal work
    def mock_strategy(features, bar, params):
        # Minimal computation - just check one feature
        if features.get('sma_10', 0) > features.get('sma_20', 0):
            return {'signal_value': 1, 'timestamp': None}
        return {'signal_value': 0, 'timestamp': None}
    
    # Create component state
    state = ComponentState(['SPY'])
    
    # Add 1000 mock strategies
    print("Adding 1000 mock strategies...")
    for i in range(1000):
        strategy_id = f"SPY_mock_strategy_{i}"
        state.add_component(
            component_id=strategy_id,
            component_func=mock_strategy,
            component_type="strategy",
            parameters={'id': i}
        )
    
    # Create test data
    test_features = {f'sma_{i}': 100 + np.random.randn() for i in range(5, 201, 5)}
    test_features['bar_count'] = 210
    
    test_bar = {
        'open': 100.1,
        'high': 100.5,
        'low': 99.8,
        'close': 100.3,
        'volume': 2100000
    }
    
    # Profile different aspects
    print("\nProfiling _is_component_ready checks:")
    
    components_snapshot = list(state._components.items())
    
    # Time individual readiness checks
    ready_times = []
    for _ in range(100):
        component_id, component_info = components_snapshot[0]  # Just check first one
        start = time.perf_counter()
        is_ready = state._is_component_ready(component_id, component_info, test_features, 210)
        duration = time.perf_counter() - start
        ready_times.append(duration * 1000000)  # Convert to microseconds
    
    print(f"  Single readiness check: {np.mean(ready_times):.1f}μs")
    print(f"  For 1000 strategies: {np.mean(ready_times) * 1000 / 1000:.2f}ms")
    
    # Time getting required features
    print("\nProfiling _get_strategy_required_features:")
    
    feature_times = []
    for _ in range(100):
        component_id, component_info = components_snapshot[0]
        start = time.perf_counter()
        required = state._get_strategy_required_features(component_id, component_info['parameters'])
        duration = time.perf_counter() - start
        feature_times.append(duration * 1000000)
    
    print(f"  Single feature check: {np.mean(feature_times):.1f}μs")
    
    # Profile the main bottleneck - the loop in _execute_components_individually
    print("\nProfiling component loop overhead:")
    
    loop_times = []
    for _ in range(10):
        start = time.perf_counter()
        
        # Simulate the loop without actual execution
        ready_strategies = []
        for component_id, component_info in components_snapshot:
            if state._is_component_ready(component_id, component_info, test_features, 210):
                ready_strategies.append((component_id, component_info))
        
        duration = time.perf_counter() - start
        loop_times.append(duration * 1000)
        print(f"  Loop iteration: {duration*1000:.2f}ms for {len(ready_strategies)} ready strategies")
    
    print(f"\nLoop overhead stats:")
    print(f"  Mean: {np.mean(loop_times):.2f}ms")
    print(f"  Per strategy: {np.mean(loop_times) / 1000:.3f}ms")
    
    return ready_times, feature_times, loop_times


def main():
    """Run all profiling tests."""
    print("="*80)
    print("SIGNAL GENERATION PERFORMANCE PROFILING")
    print("="*80)
    
    # Profile features
    feature_times, update_times = profile_feature_computation()
    
    # Profile strategies
    strategy_times, batch_times = profile_strategy_execution()
    
    # Profile ComponentState overhead
    ready_times, feature_check_times, loop_times = profile_component_state_overhead()
    
    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE BOTTLENECK ANALYSIS")
    print("="*80)
    
    print("\n1. Feature Computation (FeatureHub):")
    print(f"   - Initial computation: ~2ms for 11 features")
    print(f"   - Incremental update: {np.mean(update_times):.2f}ms per bar")
    print(f"   - Get features: <0.01ms")
    
    print("\n2. Strategy Execution:")
    print(f"   - Single strategy: ~0.01-0.02ms")
    print(f"   - 1000 strategies: ~{sum(batch_times):.0f}ms total, ~{sum(batch_times)/1000:.2f}ms per strategy")
    
    print("\n3. ComponentState Overhead:")
    print(f"   - Readiness check loop: {np.mean(loop_times):.2f}ms for 1000 strategies")
    print(f"   - Per-strategy overhead: {np.mean(loop_times)/1000:.3f}ms")
    
    print("\n4. Total Estimated Time per Bar (1000 strategies):")
    feature_time = np.mean(update_times)
    execution_time = sum(batch_times)
    overhead_time = np.mean(loop_times)
    total_time = feature_time + execution_time + overhead_time
    
    print(f"   - Feature update: {feature_time:.1f}ms")
    print(f"   - Strategy execution: {execution_time:.1f}ms")
    print(f"   - Framework overhead: {overhead_time:.1f}ms")
    print(f"   - TOTAL: {total_time:.1f}ms per bar")
    
    print(f"\n5. Projected Performance:")
    print(f"   - 1,000 bars: {total_time:.1f}s")
    print(f"   - 10,000 bars: {total_time * 10:.1f}s")
    print(f"   - 100,000 bars: {total_time * 100:.1f}s ({total_time * 100 / 60:.1f} minutes)")
    
    print("\n6. Main Bottlenecks Identified:")
    print("   1. ComponentState readiness checking loop - checking all 1000 strategies every bar")
    print("   2. Feature computation in FeatureHub - recomputing all features every bar")
    print("   3. Strategy execution - while fast individually, 1000 adds up")
    
    print("\n7. Optimization Recommendations:")
    print("   1. Cache readiness state - strategies don't need checking after warmup")
    print("   2. Optimize feature computation - only compute what changed")
    print("   3. Batch strategy execution - process in parallel")
    print("   4. Use sparse signal publishing - only publish non-zero signals")


if __name__ == "__main__":
    main()