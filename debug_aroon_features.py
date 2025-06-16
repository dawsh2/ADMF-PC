#!/usr/bin/env python3
"""Debug exactly what features are being provided to aroon_oscillator strategy."""

# Let's create a custom debug strategy that shows us what features are available
import sys
import os
sys.path.append('.')

from typing import Dict, Any, Optional
from src.core.components.discovery import strategy

@strategy(
    name='debug_aroon_features',
    feature_config=['aroon'],  # Same as aroon_oscillator
    param_feature_mapping={
        'aroon_period': 'aroon_{aroon_period}'
    }
)
def debug_aroon_features_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Debug strategy to show what features are available."""
    
    aroon_period = params.get('aroon_period', 14)
    
    print(f"\\n=== DEBUG AROON FEATURES ===")
    print(f"Bar timestamp: {bar.get('timestamp')}")
    print(f"Bar data: close={bar.get('close')}, high={bar.get('high')}, low={bar.get('low')}")
    print(f"Aroon period: {aroon_period}")
    print(f"Total features available: {len(features)}")
    print(f"Feature keys: {list(features.keys())}")
    
    # Check what aroon features are available
    for key, value in features.items():
        if 'aroon' in key.lower():
            print(f"  Aroon feature '{key}': {value} (type: {type(value)})")
            if isinstance(value, dict):
                print(f"    Dict keys: {list(value.keys())}")
                for k, v in value.items():
                    print(f"      {k}: {v}")
    
    # Try to get the specific aroon feature the strategy would look for
    expected_key = f'aroon_{aroon_period}'
    aroon_data = features.get(expected_key)
    print(f"\\nExpected key '{expected_key}': {aroon_data}")
    
    if aroon_data:
        print(f"  Type: {type(aroon_data)}")
        if isinstance(aroon_data, dict):
            print(f"  Keys: {list(aroon_data.keys())}")
            up = aroon_data.get('up')
            down = aroon_data.get('down')  
            osc = aroon_data.get('oscillator')
            print(f"  up: {up}, down: {down}, oscillator: {osc}")
    
    # Always return a signal so we know the strategy is being called
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': 0,  # Neutral signal
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'debug_aroon_features',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'debug': True,
            'features_count': len(features),
            'aroon_data_available': aroon_data is not None
        }
    }

if __name__ == "__main__":
    print("Debug strategy defined. Use this by importing in a test script.")