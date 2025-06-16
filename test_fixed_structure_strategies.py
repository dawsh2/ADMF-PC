#!/usr/bin/env python3

"""
Test the fixed structure strategies with simulated feature data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('src'))

from strategy.strategies.indicators.structure import (
    pivot_channel_breaks, pivot_channel_bounces, 
    trendline_breaks, trendline_bounces
)

def test_pivot_channel_breaks():
    print("=== Testing pivot_channel_breaks ===")
    
    # Simulate decomposed feature data as FeatureHub would provide
    features = {
        'pivot_points_standard_r1': 103.0,
        'pivot_points_standard_s1': 99.0,
        'pivot_points_standard_pivot': 101.0,
        'support_resistance_20_resistance': 104.0,
        'support_resistance_20_support': 98.0,
    }
    
    bar = {
        'close': 105.0,  # Above resistance
        'symbol': 'TEST',
        'timeframe': '5m',
        'timestamp': '2024-01-01T10:00:00Z'
    }
    
    params = {
        'pivot_type': 'standard',
        'sr_period': 20,
        'breakout_threshold': 0.001
    }
    
    try:
        result = pivot_channel_breaks(features, bar, params)
        print(f"Result: {result}")
        if result:
            print(f"Signal: {result['signal_value']}")
        else:
            print("No signal generated")
    except Exception as e:
        print(f"Error: {e}")

def test_pivot_channel_bounces():
    print("\n=== Testing pivot_channel_bounces ===")
    
    # Simulate decomposed feature data as FeatureHub would provide
    features = {
        'support_resistance_20_resistance': 104.0,
        'support_resistance_20_support': 98.0,
    }
    
    bar = {
        'close': 98.1,  # Near support
        'high': 98.5,
        'low': 97.9,
        'symbol': 'TEST',
        'timeframe': '5m',
        'timestamp': '2024-01-01T10:00:00Z'
    }
    
    params = {
        'sr_period': 20,
        'min_touches': 2,
        'bounce_threshold': 0.002
    }
    
    try:
        result = pivot_channel_bounces(features, bar, params)
        print(f"Result: {result}")
        if result:
            print(f"Signal: {result['signal_value']}")
        else:
            print("No signal generated")
    except Exception as e:
        print(f"Error: {e}")

def test_trendline_breaks():
    print("\n=== Testing trendline_breaks ===")
    
    # Simulate decomposed feature data as FeatureHub would provide
    features = {
        'trendlines_20_2_0.002_valid_uptrends': 1,
        'trendlines_20_2_0.002_valid_downtrends': 3,
        'trendlines_20_2_0.002_nearest_support': 100.0,
        'trendlines_20_2_0.002_nearest_resistance': 105.0,
    }
    
    bar = {
        'close': 99.5,  # Below support
        'symbol': 'TEST',
        'timeframe': '5m',
        'timestamp': '2024-01-01T10:00:00Z'
    }
    
    params = {
        'pivot_lookback': 20,
        'tolerance': 0.002
    }
    
    try:
        result = trendline_breaks(features, bar, params)
        print(f"Result: {result}")
        if result:
            print(f"Signal: {result['signal_value']}")
        else:
            print("No signal generated")
    except Exception as e:
        print(f"Error: {e}")

def test_trendline_bounces():
    print("\n=== Testing trendline_bounces ===")
    
    # Simulate decomposed feature data as FeatureHub would provide
    features = {
        'trendlines_20_3_0.002_nearest_support': 100.0,
        'trendlines_20_3_0.002_nearest_resistance': 105.0,
        'trendlines_20_3_0.002_valid_uptrends': 3,
        'trendlines_20_3_0.002_valid_downtrends': 1,
    }
    
    bar = {
        'close': 100.1,  # Near support
        'symbol': 'TEST',
        'timeframe': '5m',
        'timestamp': '2024-01-01T10:00:00Z'
    }
    
    params = {
        'pivot_lookback': 20,
        'min_touches': 3,
        'tolerance': 0.002
    }
    
    try:
        result = trendline_bounces(features, bar, params)
        print(f"Result: {result}")
        if result:
            print(f"Signal: {result['signal_value']}")
        else:
            print("No signal generated")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pivot_channel_breaks()
    test_pivot_channel_bounces()
    test_trendline_breaks()
    test_trendline_bounces()