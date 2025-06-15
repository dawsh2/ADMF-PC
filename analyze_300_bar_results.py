#!/usr/bin/env python3
"""
Analyze results from 300 bar run
"""
import os
import json
from collections import defaultdict

workspace = "workspaces/expansive_grid_search_f8719c17"

# Load metadata
with open(f"{workspace}/metadata.json", "r") as f:
    metadata = json.load(f)

print(f"=== RUN SUMMARY ===")
print(f"Total bars processed: {metadata.get('total_bars', 'N/A')}")
print(f"Total signals generated: {metadata.get('total_signals', 'N/A'):,}")
print(f"Total classifications: {metadata.get('total_classifications', 'N/A'):,}")
print(f"Stored changes: {metadata.get('stored_changes', 'N/A'):,}")
print(f"Compression ratio: {metadata.get('compression_ratio', 'N/A'):.1f}x")

# Count components
total_strategies = 0
total_classifiers = 0
strategies_with_signals = []
classifiers_with_classifications = []

for component_id, component_data in metadata['components'].items():
    if component_data['component_type'] == 'strategy':
        total_strategies += 1
        if 'signal_file_path' in component_data:
            strategies_with_signals.append(component_id)
    elif component_data['component_type'] == 'classifier':
        total_classifiers += 1
        if 'classification_file_path' in component_data:
            classifiers_with_classifications.append(component_id)

print(f"\n=== COMPONENT EXECUTION ===")
print(f"Total strategies configured: {total_strategies}")
print(f"Strategies that generated signals: {len(strategies_with_signals)}")
print(f"Strategy success rate: {len(strategies_with_signals)/total_strategies*100:.1f}%")

print(f"\nTotal classifiers configured: {total_classifiers}")
print(f"Classifiers that generated classifications: {len(classifiers_with_classifications)}")

# Analyze by strategy type
strategy_types = defaultdict(lambda: {'total': 0, 'with_signals': 0})
for component_id, component_data in metadata['components'].items():
    if component_data['component_type'] == 'strategy':
        strategy_type = component_data.get('strategy_type', 'unknown').replace('_grid', '')
        strategy_types[strategy_type]['total'] += 1
        if component_id in strategies_with_signals:
            strategy_types[strategy_type]['with_signals'] += 1

print("\n=== STRATEGY EXECUTION BY TYPE ===")
executed_types = []
failed_types = []

for strategy_type, counts in sorted(strategy_types.items()):
    success_rate = counts['with_signals'] / counts['total'] * 100 if counts['total'] > 0 else 0
    if counts['with_signals'] > 0:
        executed_types.append(strategy_type)
        print(f"✓ {strategy_type}: {counts['with_signals']}/{counts['total']} ({success_rate:.0f}%)")
    else:
        failed_types.append(strategy_type)

if failed_types:
    print("\nStrategies that FAILED to generate signals:")
    for ft in failed_types:
        counts = strategy_types[ft]
        print(f"✗ {ft}: 0/{counts['total']} (0%)")

# Expected vs actual
expected_strategies = [
    'sma_crossover', 'ema_crossover', 'ema_sma_crossover', 'dema_crossover', 'dema_sma_crossover',
    'tema_sma_crossover', 'macd_crossover', 'stochastic_crossover', 'vortex_crossover',
    'ichimoku_cloud_position', 'rsi_threshold', 'rsi_bands', 'cci_threshold', 'cci_bands',
    'stochastic_rsi', 'williams_r', 'roc_threshold', 'ultimate_oscillator',
    'bollinger_breakout', 'donchian_breakout', 'keltner_breakout',
    'obv_trend', 'mfi_bands', 'vwap_deviation', 'chaikin_money_flow', 'accumulation_distribution',
    'adx_trend_strength', 'parabolic_sar', 'aroon_crossover', 'supertrend', 'linear_regression_slope',
    'pivot_points', 'fibonacci_retracement', 'support_resistance_breakout', 'atr_channel_breakout', 'price_action_swing'
]

found_types = set(st.replace('_grid', '') for st in strategy_types.keys())
missing_types = set(expected_strategies) - found_types

print(f"\n=== COVERAGE ANALYSIS ===")
print(f"Expected strategy types: {len(expected_strategies)}")
print(f"Strategy types that executed: {len(executed_types)}")
print(f"Coverage: {len(executed_types)/len(expected_strategies)*100:.1f}%")

if missing_types:
    print(f"\nStrategy types completely missing (not even attempted):")
    for mt in sorted(missing_types):
        print(f"  - {mt}")

# If all 888 strategies are configured, what's the breakdown?
if total_strategies == 888:
    print("\n✅ ALL 888 STRATEGIES WERE CONFIGURED!")
    print(f"Successfully generated signals: {len(strategies_with_signals)}")
    print(f"Failed to generate signals: {888 - len(strategies_with_signals)}")
    print(f"Overall success rate: {len(strategies_with_signals)/888*100:.1f}%")
else:
    print(f"\n⚠️  Only {total_strategies} out of 888 expected strategies were configured")

# Check file sizes
print("\n=== STORAGE EFFICIENCY ===")
total_size = 0
signal_files = 0
signals_dir = f"{workspace}/traces/SPY_1m/signals"
if os.path.exists(signals_dir):
    for root, dirs, files in os.walk(signals_dir):
        for file in files:
            if file.endswith('.parquet'):
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                signal_files += 1

if signal_files > 0:
    print(f"Total signal files: {signal_files}")
    print(f"Total storage size: {total_size/1024/1024:.2f} MB")
    print(f"Average file size: {total_size/signal_files/1024:.1f} KB")
    print(f"Storage per strategy: {total_size/len(strategies_with_signals)/1024:.1f} KB")