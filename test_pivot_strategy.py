#!/usr/bin/env python3
"""
Test pivot_points strategy directly
"""
from src.strategy.strategies.indicators.structure import pivot_points
from src.strategy.components.features.hub import FeatureHub, compute_feature
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range('2023-01-01', periods=100, freq='1min')
test_df = pd.DataFrame({
    'open': np.random.uniform(99, 101, 100),
    'high': np.random.uniform(100, 102, 100),
    'low': np.random.uniform(98, 100, 100),
    'close': np.random.uniform(99, 101, 100),
    'volume': np.random.uniform(900000, 1100000, 100)
}, index=dates)

# Create a test bar
test_bar = {
    'open': 100.0,
    'high': 101.0, 
    'low': 99.0,
    'close': 100.5,
    'volume': 1000000,
    'timestamp': '2023-01-01 10:00:00',
    'symbol': 'TEST',
    'timeframe': '1m'
}

print("Testing pivot_points strategy...")

# Check strategy metadata
if hasattr(pivot_points, '_strategy_metadata'):
    metadata = pivot_points._strategy_metadata
    print(f"Strategy metadata: {metadata}")
    feature_config = metadata.get('feature_config', {})
    print(f"Required features: {list(feature_config.keys())}")
else:
    print("No strategy metadata found!")

# Try to compute required features
try:
    # Compute pivot points feature
    pivot_result = compute_feature('pivot_points', test_df, pivot_type='standard')
    print(f"\nPivot points feature result type: {type(pivot_result)}")
    
    if isinstance(pivot_result, dict):
        print("Feature components:")
        for key, value in pivot_result.items():
            if hasattr(value, 'iloc'):
                print(f"  {key}: {value.iloc[-1] if len(value) > 0 else 'No data'}")
            else:
                print(f"  {key}: {value}")
    
    # Create features dict as the strategy expects
    features = {}
    if isinstance(pivot_result, dict):
        for key, series in pivot_result.items():
            if hasattr(series, 'iloc') and len(series) > 0:
                features[f'pivot_points_standard_{key}'] = float(series.iloc[-1])
    
    print(f"\nFeatures prepared: {list(features.keys())}")
    
    # Test strategy execution
    params = {'pivot_type': 'standard'}
    result = pivot_points(features, test_bar, params)
    
    if result:
        print(f"\n✓ Strategy executed successfully!")
        print(f"Signal value: {result.get('signal_value')}")
        print(f"Metadata: {result.get('metadata')}")
    else:
        print(f"\n✗ Strategy returned None")
        print(f"Features provided: {features}")
        
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()