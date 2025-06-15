#!/usr/bin/env python3
"""
Count signal generation results
"""
import os
import json
from collections import defaultdict

workspace = "workspaces/expansive_grid_search_5fe966d1"

# Load metadata
with open(f"{workspace}/metadata.json", "r") as f:
    metadata = json.load(f)

# Count components
total_strategies = 0
total_classifiers = 0
strategies_with_signals = []
classifiers_with_classifications = []

for component_id, component_data in metadata['components'].items():
    if component_data['component_type'] == 'strategy':
        total_strategies += 1
        if 'signal_file_path' in component_data and os.path.exists(f"{workspace}/{component_data['signal_file_path']}"):
            strategies_with_signals.append(component_id)
    elif component_data['component_type'] == 'classifier':
        total_classifiers += 1
        if 'classification_file_path' in component_data:
            classifiers_with_classifications.append(component_id)

print(f"Total strategies: {total_strategies}")
print(f"Strategies that generated signals: {len(strategies_with_signals)}")
print(f"Success rate: {len(strategies_with_signals)/total_strategies*100:.1f}%")

print(f"\nTotal classifiers: {total_classifiers}")
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
for strategy_type, counts in sorted(strategy_types.items()):
    success_rate = counts['with_signals'] / counts['total'] * 100 if counts['total'] > 0 else 0
    print(f"{strategy_type}: {counts['with_signals']}/{counts['total']} ({success_rate:.0f}%)")

# Check signal statistics from metadata
print(f"\n=== SIGNAL STATISTICS ===")
print(f"Total bars processed: {metadata.get('total_bars', 'N/A')}")
print(f"Total signals generated: {metadata.get('total_signals', 'N/A'):,}")
print(f"Total signal changes stored: {metadata.get('stored_changes', 'N/A'):,}")
print(f"Compression ratio: {metadata.get('compression_ratio', 'N/A'):.1f}x")

# Find strategies that didn't generate signals
print("\n=== MISSING STRATEGIES ===")
# Expected strategies from config
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

if missing_types:
    print(f"Strategy types that generated NO signals at all:")
    for mt in sorted(missing_types):
        print(f"  - {mt}")
else:
    print("All expected strategy types generated at least some signals!")

# Show sample of high-frequency strategies
print("\n=== HIGH FREQUENCY STRATEGIES ===")
high_freq = []
for component_id, component_data in metadata['components'].items():
    if component_data['component_type'] == 'strategy' and 'signal_frequency' in component_data:
        freq = component_data['signal_frequency']
        if freq > 0.1:  # More than 10% of bars had signal changes
            high_freq.append((component_id, freq))

high_freq.sort(key=lambda x: x[1], reverse=True)
for strat_id, freq in high_freq[:10]:
    print(f"  {strat_id}: {freq*100:.1f}% signal change frequency")