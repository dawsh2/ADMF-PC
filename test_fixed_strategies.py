#!/usr/bin/env python3
"""Test that fixed strategies generate actual signal changes."""

import sys
sys.path.append('.')

from datetime import datetime
import importlib

# Import modules
from src.strategy.components.features.hub import FeatureHub
from src.core.components.discovery import get_component_registry

# Import strategy modules
importlib.import_module('src.strategy.strategies.indicators.trend')
importlib.import_module('src.strategy.strategies.indicators.volatility')
importlib.import_module('src.strategy.strategies.indicators.momentum')

# Create feature hub
hub = FeatureHub(['SPY'])
feature_configs = {
    'psar_0.02_0.2': {'type': 'psar', 'af_start': 0.02, 'af_max': 0.2},
    'supertrend_10_3.0': {'type': 'supertrend', 'period': 10, 'multiplier': 3.0},
    'adx_14': {'type': 'adx', 'period': 14},
    'aroon_25': {'type': 'aroon', 'period': 25},
}
hub.configure_features(feature_configs)

# Get strategies
registry = get_component_registry()
strategies_to_test = ['parabolic_sar', 'supertrend', 'adx_trend_strength', 'aroon_crossover']

# Generate trending market data that should trigger signals
bars = []
# First 50 bars - uptrend
for i in range(50):
    price = 100 + i * 0.5
    bars.append({
        'symbol': 'SPY',
        'timestamp': f'2024-01-01 09:{30+i:02d}',
        'open': price - 0.3,
        'high': price + 0.5,
        'low': price - 0.5,
        'close': price,
        'volume': 100000
    })

# Next 50 bars - downtrend
for i in range(50):
    price = 125 - i * 0.5
    bars.append({
        'symbol': 'SPY',
        'timestamp': f'2024-01-01 10:{30+i:02d}',
        'open': price + 0.3,
        'high': price + 0.5,
        'low': price - 0.5,
        'close': price,
        'volume': 100000
    })

# Process bars and track signals
print("Testing fixed strategies with trending data:\n")

for strategy_name in strategies_to_test:
    print(f"\n{'='*60}")
    print(f"Testing {strategy_name}")
    print('='*60)
    
    strategy_info = registry.get_component(strategy_name)
    if not strategy_info:
        print(f"  ❌ Strategy not found!")
        continue
        
    # Reset hub for clean test
    hub.reset()
    
    # Track signal changes
    last_signal = None
    signal_changes = []
    
    # Process all bars
    for i, bar in enumerate(bars):
        hub.update_bar('SPY', bar)
        features = hub.get_features('SPY')
        
        # Get appropriate params for each strategy
        if strategy_name == 'parabolic_sar':
            params = {'af_start': 0.02, 'af_max': 0.2}
        elif strategy_name == 'supertrend':
            params = {'period': 10, 'multiplier': 3.0}
        elif strategy_name == 'adx_trend_strength':
            params = {'adx_period': 14, 'di_period': 14, 'adx_threshold': 25}
        elif strategy_name == 'aroon_crossover':
            params = {'period': 25}
        else:
            params = {}
            
        signal = strategy_info.factory(features, bar, params)
        
        if signal and signal.get('signal_value') != last_signal:
            last_signal = signal.get('signal_value')
            signal_changes.append({
                'bar': i + 1,
                'price': bar['close'],
                'signal': last_signal,
                'timestamp': bar['timestamp']
            })
            
        # Print status every 25 bars
        if i == 24 or i == 49 or i == 74 or i == 99:
            print(f"  Bar {i+1}: price={bar['close']:.2f}, features={len(features)}, signal={signal.get('signal_value') if signal else 'None'}")
    
    print(f"\n  Signal changes: {len(signal_changes)}")
    for change in signal_changes[:5]:  # Show first 5 changes
        print(f"    Bar {change['bar']}: {change['signal']} at price {change['price']:.2f}")
    if len(signal_changes) > 5:
        print(f"    ... and {len(signal_changes) - 5} more changes")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print("If strategies show 0 signal changes, they're returning constant signals.")
print("The tracer only stores signal CHANGES, not constant signals.")