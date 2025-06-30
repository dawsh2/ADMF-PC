#!/usr/bin/env python3
"""Minimal test to debug signal generation."""

import pandas as pd
import numpy as np
from datetime import datetime

# Import what we need
from src.strategy.strategies.indicators.volatility import bollinger_bands
from src.strategy.components.features.hub import FeatureHub
from src.strategy.components.features.indicators.volatility import BollingerBands

# Load data
df = pd.read_csv('data/SPY_5m.csv')
print(f"Loaded {len(df)} bars")

# Create FeatureHub
hub = FeatureHub()

# Configure features - this is what was missing!
feature_configs = {
    'bollinger_bands_20_2.0_upper': {
        'type': 'bollinger_bands',
        'period': 20,
        'std_dev': 2.0,
        'component': 'upper'
    },
    'bollinger_bands_20_2.0_middle': {
        'type': 'bollinger_bands',
        'period': 20,
        'std_dev': 2.0,
        'component': 'middle'
    },
    'bollinger_bands_20_2.0_lower': {
        'type': 'bollinger_bands',
        'period': 20,
        'std_dev': 2.0,
        'component': 'lower'
    }
}
hub.configure_features(feature_configs)

# Process bars to calculate features
symbol = 'SPY'
signals = []

for i in range(100):
    bar_data = df.iloc[i]
    
    # Update feature hub
    bar_dict = {
        'open': bar_data['open'],
        'high': bar_data['high'],
        'low': bar_data['low'],
        'close': bar_data['close'],
        'volume': bar_data['volume']
    }
    
    hub.update_bar(symbol, bar_dict)
    
    # After warmup, check for signals
    if i >= 25:  # 20 bars for BB + some buffer
        features = hub.get_features(symbol)
        
        # Look for bollinger band features
        bb_features = {k: v for k, v in features.items() if 'bollinger' in k and '20_2.0' in k}
        
        if bb_features:
            # Create a complete feature set with the expected keys
            test_features = {
                'bollinger_bands_20_2.0_upper': bb_features.get('bollinger_bands_20_2.0_upper'),
                'bollinger_bands_20_2.0_middle': bb_features.get('bollinger_bands_20_2.0_middle'),
                'bollinger_bands_20_2.0_lower': bb_features.get('bollinger_bands_20_2.0_lower')
            }
            
            # Check if all required features are present
            if all(v is not None for v in test_features.values()):
                # Call strategy
                bar_for_strategy = {
                    'close': bar_data['close'],
                    'symbol': symbol,
                    'timeframe': '5m',
                    'timestamp': bar_data['timestamp'],
                    'open': bar_data['open'],
                    'high': bar_data['high'],
                    'low': bar_data['low'],
                    'volume': bar_data['volume']
                }
                
                params = {'period': 20, 'std_dev': 2.0}
                
                try:
                    signal = bollinger_bands(test_features, bar_for_strategy, params)
                    if signal and signal.get('signal_value') != 0:
                        signals.append((i, signal))
                        print(f"\nBar {i}: SIGNAL GENERATED!")
                        print(f"  Close: {bar_data['close']:.2f}")
                        print(f"  Upper: {test_features['bollinger_bands_20_2.0_upper']:.2f}")
                        print(f"  Middle: {test_features['bollinger_bands_20_2.0_middle']:.2f}")
                        print(f"  Lower: {test_features['bollinger_bands_20_2.0_lower']:.2f}")
                        print(f"  Signal: {signal}")
                except Exception as e:
                    print(f"Error calling strategy at bar {i}: {e}")
            else:
                print(f"Bar {i}: Missing features - {[k for k, v in test_features.items() if v is None]}")
        else:
            print(f"Bar {i}: No bollinger features found. Available: {list(features.keys())[:5]}")

print(f"\n=== Summary ===")
print(f"Total signals: {len(signals)}")
for bar_num, signal in signals[:10]:  # Show first 10
    print(f"  Bar {bar_num}: {signal['signal_value']} at price {signal['metadata']['price']:.2f}")