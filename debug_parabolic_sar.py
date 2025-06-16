#!/usr/bin/env python3
"""Debug why parabolic_sar isn't generating signals."""

import sys
sys.path.append('.')

from src.core.components.discovery import get_component_registry
from src.strategy.components.features.hub import FeatureHub
import importlib

# Import strategy modules
importlib.import_module('src.strategy.strategies.indicators.trend')

# Get the strategy
registry = get_component_registry()
psar_info = registry.get_component('parabolic_sar')

if psar_info:
    print("✓ parabolic_sar strategy is registered")
    print(f"  Factory: {psar_info.factory}")
    print(f"  Features: {psar_info.metadata.get('feature_config')}")
    print(f"  Param mapping: {psar_info.metadata.get('param_feature_mapping')}")
    
    # Test the strategy with mock data
    print("\nTesting strategy execution...")
    
    # Create a feature hub and configure it
    hub = FeatureHub(['SPY'])
    hub.configure_features({
        'psar_0.02_0.2': {'type': 'psar', 'af_start': 0.02, 'af_max': 0.2}
    })
    
    # Simulate some bars
    bars = [
        {'symbol': 'SPY', 'timestamp': '2024-01-01 09:30', 'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'volume': 1000},
        {'symbol': 'SPY', 'timestamp': '2024-01-01 09:31', 'open': 100.5, 'high': 102, 'low': 100, 'close': 101.5, 'volume': 1100},
        {'symbol': 'SPY', 'timestamp': '2024-01-01 09:32', 'open': 101.5, 'high': 102.5, 'low': 101, 'close': 102, 'volume': 1200},
    ]
    
    for i, bar in enumerate(bars):
        hub.update_bar('SPY', bar)
        features = hub.get_features('SPY')
        print(f"\nBar {i+1}: close={bar['close']}")
        print(f"  Features: {features}")
        
        # Call the strategy
        params = {'af_start': 0.02, 'af_max': 0.2}
        signal = psar_info.factory(features, bar, params)
        print(f"  Signal: {signal}")
else:
    print("✗ parabolic_sar strategy NOT found")