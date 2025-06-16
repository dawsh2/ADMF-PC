#!/usr/bin/env python3
"""Test all strategies to see which ones are actually implemented and working."""

import sys
sys.path.append('.')

from src.core.components.discovery import get_component_registry
import importlib

# Import all strategy modules to ensure decorators run
modules = [
    'src.strategy.strategies.indicators.trend',
    'src.strategy.strategies.indicators.momentum', 
    'src.strategy.strategies.indicators.oscillators',
    'src.strategy.strategies.indicators.volatility',
    'src.strategy.strategies.indicators.volume',
    'src.strategy.strategies.indicators.structure',
    'src.strategy.strategies.indicators.crossovers',
    'src.strategy.strategies.ensemble.trend_momentum_composite',
    'src.strategy.strategies.ensemble.structure_confluence',
]

for module in modules:
    try:
        importlib.import_module(module)
    except ImportError as e:
        print(f"Could not import {module}: {e}")

# Get all registered strategies
registry = get_component_registry()
all_component_names = registry.list_components()
strategies = []
for name in all_component_names:
    info = registry.get_component(name)
    if info and info.component_type == 'strategy':
        strategies.append(info)

print(f"\nTotal registered strategies: {len(strategies)}")
print("\nRegistered strategy names:")
for s in sorted(strategies, key=lambda x: x.name):
    print(f"  - {s.name}")

# Expected strategies from config
expected = [
    'sma_crossover', 'ema_crossover', 'ema_sma_crossover',
    'dema_sma_crossover', 'tema_sma_crossover', 'vwma_sma_crossover',
    'macd_crossover', 'rsi_threshold', 'rsi_bands', 
    'stochastic_threshold', 'williams_r_threshold', 'cci_threshold',
    'cci_bands', 'mfi_bands', 'ultimate_oscillator_threshold',
    'bollinger_breakout', 'keltner_breakout', 'donchian_breakout',
    'supertrend', 'atr_channel_breakout', 'vwap_distance',
    'obv_trend', 'chaikin_money_flow', 'volume_momentum',
    'adx_threshold', 'aroon_crossover', 'vortex_crossover',
    'pivot_bounce', 'swing_high_low', 'linear_regression_slope',
    'structure_confluence', 'ichimoku_trend', 'parabolic_sar',
    'hma_trend', 'wma_trend', 'dema_trend',
    'tema_trend', 'trend_strength', 'momentum_rotation', 
    'roc_threshold', 'volume_ratio_threshold'
]

# Find missing
registered_names = {s.name for s in strategies}
missing = sorted(set(expected) - registered_names)

print(f"\nMissing {len(missing)} expected strategies:")
for m in missing:
    print(f"  - {m}")

# Find similar names that might be the actual implementation
print("\nPossible matches for missing strategies:")
for m in missing:
    # Look for similar names
    similar = [s.name for s in strategies if m.replace('_', '') in s.name.replace('_', '') or s.name.replace('_', '') in m.replace('_', '')]
    if similar:
        print(f"  {m} -> {similar}")