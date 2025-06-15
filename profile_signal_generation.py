#!/usr/bin/env python3
"""
Profile signal generation performance for 1k strategies.

Adds timing measurements to identify bottlenecks.
"""

import logging
from datetime import datetime
import time
from collections import defaultdict
import numpy as np
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Monkey patch the ComponentState and FeatureHub to add timing
timing_data = defaultdict(list)

def timed_section(section_name: str):
    """Decorator to time a section of code."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            timing_data[section_name].append(duration)
            return result
        return wrapper
    return decorator

# Patch ComponentState._execute_components_individually
from src.strategy.state import ComponentState
from src.strategy.components.features.hub import FeatureHub

original_execute = ComponentState._execute_components_individually
original_update_features = FeatureHub._update_features
original_get_features = ComponentState.get_features
original_publish_signal = ComponentState._publish_signal
original_process_output = ComponentState._process_component_output

def patched_execute_components_individually(self, symbol: str, features: Dict[str, Any], 
                                          bar: Dict[str, float], timestamp: datetime) -> None:
    """Patched version with timing."""
    current_bars = self._bar_count.get(symbol, 0)
    
    # Time the entire execution
    start_total = time.perf_counter()
    
    # Time component readiness check
    start_readiness = time.perf_counter()
    components_snapshot = list(self._components.items())
    ready_classifiers = []
    ready_strategies = []
    
    for component_id, component_info in components_snapshot:
        component_type = component_info['component_type']
        if self._is_component_ready(component_id, component_info, features, current_bars):
            if component_type == 'classifier':
                ready_classifiers.append((component_id, component_info))
            else:
                ready_strategies.append((component_id, component_info))
    
    readiness_time = time.perf_counter() - start_readiness
    timing_data['readiness_check'].append(readiness_time)
    
    if current_bars % 10 == 0:
        logger.info(f"Bar {current_bars}: {len(ready_strategies)} strategies ready out of {len(components_snapshot)} total (readiness check: {readiness_time*1000:.2f}ms)")
    
    # Time classifier execution
    start_classifiers = time.perf_counter()
    current_classifications = {}
    outputs_to_update = {}
    
    for component_id, component_info in ready_classifiers:
        try:
            result = component_info['function'](
                features=features,
                params=component_info['parameters']
            )
            
            if result:
                outputs_to_update[component_id] = result
                self._process_component_output(
                    component_id=component_id,
                    component_type=component_info['component_type'],
                    result=result,
                    symbol=symbol,
                    timestamp=timestamp,
                    component_info=component_info
                )
                
                if result.get('regime'):
                    classifier_name = component_id.split('_', 1)[1] if '_' in component_id else component_id
                    current_classifications[f'regime_{classifier_name}'] = result['regime']
                    current_classifications[f'regime_confidence_{classifier_name}'] = result.get('confidence', 0.0)
                    
        except Exception as e:
            logger.error(f"Error executing classifier {component_id}: {e}")
    
    classifiers_time = time.perf_counter() - start_classifiers
    timing_data['classifier_execution'].append(classifiers_time)
    
    # Add classifications to features
    if current_classifications:
        features = features.copy()
        features.update(current_classifications)
    
    # Time strategy execution
    start_strategies = time.perf_counter()
    strategy_count = 0
    signal_count = 0
    
    for component_id, component_info in ready_strategies:
        try:
            start_single = time.perf_counter()
            result = component_info['function'](
                features=features,
                bar=bar,
                params=component_info['parameters']
            )
            single_time = time.perf_counter() - start_single
            timing_data['single_strategy_execution'].append(single_time)
            
            if result:
                strategy_count += 1
                signal_value = result.get('signal_value', 0)
                if signal_value != 0:
                    signal_count += 1
                
                outputs_to_update[component_id] = result
                self._process_component_output(
                    component_id=component_id,
                    component_type=component_info['component_type'],
                    result=result,
                    symbol=symbol,
                    timestamp=timestamp,
                    component_info=component_info
                )
                
        except Exception as e:
            logger.error(f"Error executing strategy {component_id}: {e}", exc_info=True)
    
    strategies_time = time.perf_counter() - start_strategies
    timing_data['strategy_execution'].append(strategies_time)
    
    # Time output update
    start_update = time.perf_counter()
    for component_id, result in outputs_to_update.items():
        if component_id in self._components:
            self._components[component_id]['last_output'] = result
    update_time = time.perf_counter() - start_update
    timing_data['output_update'].append(update_time)
    
    total_time = time.perf_counter() - start_total
    timing_data['total_execution'].append(total_time)
    
    if current_bars % 10 == 0:
        logger.info(f"Bar {current_bars} timing: total={total_time*1000:.2f}ms, "
                   f"strategies={strategies_time*1000:.2f}ms ({strategy_count} strategies, {signal_count} signals), "
                   f"avg_per_strategy={(strategies_time/max(1, len(ready_strategies)))*1000:.2f}ms")

def patched_update_features(self, symbol: str) -> None:
    """Patched version of _update_features with timing."""
    start_total = time.perf_counter()
    
    # Time data preparation
    start_prep = time.perf_counter()
    data_dict = {}
    for field, deque_data in self.price_data[symbol].items():
        if len(deque_data) > 0:
            data_dict[field] = list(deque_data)
    
    if not data_dict or len(data_dict['close']) < 2:
        return
    
    import pandas as pd
    df = pd.DataFrame(data_dict)
    prep_time = time.perf_counter() - start_prep
    timing_data['feature_data_prep'].append(prep_time)
    
    # Time feature computation
    start_compute = time.perf_counter()
    symbol_features = {}
    
    for feature_name, config in self.feature_configs.items():
        try:
            start_single = time.perf_counter()
            feature_type = config.get('feature')
            if feature_type not in self.FEATURE_REGISTRY:
                continue
            
            feature_params = {k: v for k, v in config.items() 
                            if k not in ['feature', 'is_raw_data']}
            
            from src.strategy.components.features.hub import compute_feature
            result = compute_feature(feature_type, df, **feature_params)
            
            if isinstance(result, dict):
                for sub_name, series in result.items():
                    if len(series) > 0 and not pd.isna(series.iloc[-1]):
                        symbol_features[f"{feature_name}_{sub_name}"] = float(series.iloc[-1])
            else:
                if len(result) > 0 and not pd.isna(result.iloc[-1]):
                    symbol_features[feature_name] = float(result.iloc[-1])
            
            single_time = time.perf_counter() - start_single
            timing_data[f'feature_{feature_type}'].append(single_time)
                    
        except Exception as e:
            logger.error("Error computing feature %s for %s: %s", feature_name, symbol, e)
    
    compute_time = time.perf_counter() - start_compute
    timing_data['feature_computation'].append(compute_time)
    
    # Time cache update
    start_cache = time.perf_counter()
    self.feature_cache[symbol].update(symbol_features)
    cache_time = time.perf_counter() - start_cache
    timing_data['feature_cache_update'].append(cache_time)
    
    total_time = time.perf_counter() - start_total
    timing_data['feature_update_total'].append(total_time)
    
    bar_count = self.bar_count.get(symbol, 0)
    if bar_count % 10 == 0:
        logger.info(f"Feature update timing: total={total_time*1000:.2f}ms, "
                   f"compute={compute_time*1000:.2f}ms ({len(symbol_features)} features)")

def patched_get_features(self, symbol: str) -> Dict[str, Any]:
    """Patched version with timing."""
    start = time.perf_counter()
    
    if self._feature_hub:
        features = self._feature_hub.get_features(symbol).copy()
    else:
        features = {}
    
    features['bar_count'] = self._bar_count.get(symbol, 0)
    
    duration = time.perf_counter() - start
    timing_data['get_features'].append(duration)
    
    return features

def patched_publish_signal(self, signal) -> None:
    """Patched version with timing."""
    start = time.perf_counter()
    original_publish_signal(self, signal)
    duration = time.perf_counter() - start
    timing_data['publish_signal'].append(duration)

def patched_process_output(self, component_id: str, component_type: str,
                          result: Dict[str, Any], symbol: str, timestamp: datetime,
                          component_info=None) -> None:
    """Patched version with timing."""
    start = time.perf_counter()
    original_process_output(self, component_id, component_type, result, symbol, timestamp, component_info)
    duration = time.perf_counter() - start
    timing_data['process_output'].append(duration)

# Apply patches
ComponentState._execute_components_individually = patched_execute_components_individually
FeatureHub._update_features = patched_update_features
ComponentState.get_features = patched_get_features
ComponentState._publish_signal = patched_publish_signal
ComponentState._process_component_output = patched_process_output

# Patch FeatureHub.FEATURE_REGISTRY
FeatureHub.FEATURE_REGISTRY = __import__('src.strategy.components.features.hub', fromlist=['FEATURE_REGISTRY']).FEATURE_REGISTRY

def print_timing_summary():
    """Print timing summary statistics."""
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    
    for section, times in sorted(timing_data.items()):
        if times:
            times_ms = [t * 1000 for t in times]
            print(f"\n{section}:")
            print(f"  Count: {len(times)}")
            print(f"  Total: {sum(times_ms):.2f}ms")
            print(f"  Mean: {np.mean(times_ms):.2f}ms")
            print(f"  Median: {np.median(times_ms):.2f}ms")
            print(f"  Min: {min(times_ms):.2f}ms")
            print(f"  Max: {max(times_ms):.2f}ms")
            if len(times) > 1:
                print(f"  Std: {np.std(times_ms):.2f}ms")

def run_profiling_test():
    """Run a test with 20 bars and profile performance."""
    import yaml
    
    # Read the expansive grid search config
    config_path = "/Users/daws/ADMF-PC/config/expansive_grid_search.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Limit to 20 bars for profiling
    print("Setting up profiling test with 20 bars...")
    
    # Import backtest runner
    from src.core.backtest_runner import BacktestRunner
    
    # Create runner and override bar limit
    runner = BacktestRunner(config)
    
    # Patch the data adapter to limit bars
    print("Running backtest with timing measurements...")
    start_time = time.time()
    
    # Run with custom bar limit
    original_limit = config.get('backtest', {}).get('data_limit', None)
    config['backtest']['data_limit'] = 20
    runner.config = config
    
    results = runner.run()
    
    total_time = time.time() - start_time
    print(f"\nTotal backtest time: {total_time:.2f}s")
    
    # Print timing summary
    print_timing_summary()
    
    # Additional analysis
    if 'strategy_execution' in timing_data and timing_data['strategy_execution']:
        total_strategy_time = sum(timing_data['strategy_execution'])
        total_strategy_calls = len(timing_data['single_strategy_execution'])
        print(f"\nStrategy Execution Analysis:")
        print(f"  Total strategy execution time: {total_strategy_time*1000:.2f}ms")
        print(f"  Total strategy calls: {total_strategy_calls}")
        print(f"  Average time per strategy call: {(total_strategy_time/max(1, total_strategy_calls))*1000:.2f}ms")
    
    if 'feature_update_total' in timing_data and timing_data['feature_update_total']:
        total_feature_time = sum(timing_data['feature_update_total'])
        print(f"\nFeature Computation Analysis:")
        print(f"  Total feature update time: {total_feature_time*1000:.2f}ms")
        print(f"  Feature updates: {len(timing_data['feature_update_total'])}")
        
        # Show breakdown by feature type
        feature_types = [k for k in timing_data.keys() if k.startswith('feature_') and not k.endswith('_total')]
        if feature_types:
            print(f"\n  Feature type breakdown:")
            for ft in sorted(feature_types):
                if timing_data[ft]:
                    ft_times = [t * 1000 for t in timing_data[ft]]
                    print(f"    {ft.replace('feature_', '')}: "
                          f"mean={np.mean(ft_times):.2f}ms, "
                          f"total={sum(ft_times):.2f}ms, "
                          f"count={len(ft_times)}")

if __name__ == "__main__":
    print("Starting signal generation profiling...")
    print("This will run 20 bars with ~1000 strategies")
    print("-" * 80)
    
    try:
        run_profiling_test()
    except Exception as e:
        logger.error(f"Error during profiling: {e}", exc_info=True)
        print_timing_summary()  # Print what we have so far