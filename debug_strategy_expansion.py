#!/usr/bin/env python3
"""Debug strategy parameter expansion."""

import yaml
from src.core.coordinator.topology import TopologyBuilder

# Load the expansive grid search config
with open('config/expansive_grid_search.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create topology builder
builder = TopologyBuilder()

# Check original strategies
print(f"Original strategies in config: {len(config['strategies'])}")
for i, strat in enumerate(config['strategies'][:5]):
    print(f"  {i}: {strat.get('type')} - {strat.get('name')}")
print("  ...")

# Expand strategy parameters
expanded = builder._expand_strategy_parameters(config['strategies'])

print(f"\nExpanded strategies: {len(expanded)}")

# Count by type
by_type = {}
for strat in expanded:
    strat_type = strat.get('name', strat.get('type'))
    by_type[strat_type] = by_type.get(strat_type, 0) + 1

print("\nExpanded count by type:")
for strat_type, count in sorted(by_type.items()):
    print(f"  {strat_type}: {count}")

# Check which ones are missing from your trace output
missing = [
    'accumulation_distribution_grid',
    'adx_trend_strength_grid',
    'aroon_crossover_grid',
    'dema_crossover_grid',
    'donchian_breakout_grid',
    'fibonacci_retracement_grid',
    'ichimoku_grid',
    'keltner_breakout_grid',
    'linear_regression_slope_grid',
    'macd_crossover_grid',
    'obv_trend_grid',
    'parabolic_sar_grid',
    'pivot_points_grid',
    'price_action_swing_grid',
    'stochastic_crossover_grid',
    'stochastic_rsi_grid',
    'supertrend_grid',
    'support_resistance_breakout_grid',
    'ultimate_oscillator_grid',
    'vortex_crossover_grid',
    'vwap_deviation_grid',
    'williams_r_grid'
]

print("\nChecking missing strategies:")
for miss in missing:
    found = sum(1 for s in expanded if s.get('name') == miss)
    if found > 0:
        print(f"  ✓ {miss}: {found} variants")
    else:
        print(f"  ✗ {miss}: NOT FOUND")