#!/usr/bin/env python3
"""Test execution of a missing strategy directly."""

import sys
sys.path.append('.')

from datetime import datetime
import pandas as pd

# Import modules
from src.core.components.discovery import get_component_registry
from src.strategy.components.features.hub import FeatureHub
from src.strategy.state import ComponentState
from src.core.events.bus import EventBus
import importlib

# Import strategy module
importlib.import_module('src.strategy.strategies.indicators.trend')

# Create event bus
event_bus = EventBus()

# Create feature hub
feature_hub = FeatureHub(['SPY'])
feature_configs = {
    'psar_0.02_0.2': {'type': 'psar', 'af_start': 0.02, 'af_max': 0.2},
    'supertrend_10_3.0': {'type': 'supertrend', 'period': 10, 'multiplier': 3.0},
}
feature_hub.configure_features(feature_configs)

# Get strategy from registry
registry = get_component_registry()
psar_info = registry.get_component('parabolic_sar')
supertrend_info = registry.get_component('supertrend')

# Create stateless components
stateless_components = {
    'SPY_parabolic_sar_test': {
        'strategy_func': psar_info.factory,
        'strategy_id': 'SPY_parabolic_sar_test',
        'params': {'af_start': 0.02, 'af_max': 0.2}
    },
    'SPY_supertrend_test': {
        'strategy_func': supertrend_info.factory,
        'strategy_id': 'SPY_supertrend_test', 
        'params': {'period': 10, 'multiplier': 3.0}
    }
}

# Create ComponentState
print("Creating ComponentState with strategies...")
state = ComponentState(
    event_bus=event_bus,
    feature_hub=feature_hub,
    stateless_components=stateless_components
)

# Test direct strategy execution
print("\nTesting direct strategy execution:")

# Create test bars
bars = []
base_price = 100
for i in range(50):
    price = base_price + (i * 0.5)  # Upward trend
    bars.append({
        'symbol': 'SPY',
        'timestamp': datetime.now(),
        'open': price - 0.2,
        'high': price + 0.3,
        'low': price - 0.3,
        'close': price,
        'volume': 100000 + i * 1000
    })

# Process bars
for i, bar in enumerate(bars):
    # Update feature hub
    feature_hub.update_bar('SPY', bar)
    features = feature_hub.get_features('SPY')
    
    # Call strategies directly
    if i % 10 == 0:  # Print every 10th bar
        print(f"\nBar {i+1}: close={bar['close']:.2f}")
        print(f"  Features available: {list(features.keys())}")
        
        # Test parabolic_sar
        if psar_info:
            signal = psar_info.factory(features, bar, {'af_start': 0.02, 'af_max': 0.2})
            print(f"  PSAR signal: {signal}")
            
        # Test supertrend
        if supertrend_info:
            signal = supertrend_info.factory(features, bar, {'period': 10, 'multiplier': 3.0})
            print(f"  Supertrend signal: {signal}")
    
    # Also test ComponentState execution
    state.on_bar_data({
        'type': 'BAR_DATA',
        'data': bar
    })

print("\nChecking if ComponentState generated any signals...")
# The signals should have printed to console with 📡 emoji