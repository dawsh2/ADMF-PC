#!/usr/bin/env python3
"""
Test script to verify new features are working properly.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from src.strategy.components.features.hub import FEATURE_REGISTRY, compute_feature

def create_test_data(n=100):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100
    prices = []
    
    for i in range(n):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        base_price *= (1 + change)
        prices.append(base_price)
    
    # Create OHLCV data
    close = pd.Series(prices)
    high = close * (1 + np.random.uniform(0, 0.03, n))  # Up to 3% higher
    low = close * (1 - np.random.uniform(0, 0.03, n))   # Up to 3% lower
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(100000, 1000000, n))
    
    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

def test_feature(feature_name, data, **kwargs):
    """Test a specific feature."""
    try:
        result = compute_feature(feature_name, data, **kwargs)
        print(f"✅ {feature_name}: SUCCESS - {type(result).__name__}")
        if isinstance(result, dict):
            print(f"   Returns dict with keys: {list(result.keys())}")
        return True
    except Exception as e:
        print(f"❌ {feature_name}: ERROR - {str(e)}")
        return False

def main():
    print("Testing new features in FEATURE_REGISTRY...")
    print(f"Total features registered: {len(FEATURE_REGISTRY)}")
    print()
    
    # Create test data
    data = create_test_data()
    
    # Test the problematic features that were showing warnings
    problematic_features = [
        'ultimate', 'supertrend', 'psar', 'mfi', 'obv', 'roc', 
        'cmf', 'ad', 'aroon', 'vwap', 'stochastic_rsi'
    ]
    
    print("Testing previously problematic features:")
    success_count = 0
    for feature in problematic_features:
        if test_feature(feature, data):
            success_count += 1
    
    print(f"\nResults: {success_count}/{len(problematic_features)} features working")
    
    if success_count == len(problematic_features):
        print("🎉 All features are now working! The warnings should be eliminated.")
    else:
        print("⚠️  Some features still have issues. Check the errors above.")

if __name__ == "__main__":
    main()