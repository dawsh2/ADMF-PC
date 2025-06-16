#!/usr/bin/env python3

"""
Quick test to verify that the fixed strategies would generate signals with proper data.
"""

import sys
sys.path.insert(0, 'src')

# Test individual strategy functions directly
def test_strategies():
    print("Testing fixed structure strategies...")
    
    # Test pivot_channel_breaks
    try:
        from src.strategy.strategies.indicators.structure import pivot_channel_breaks
        
        features = {
            'pivot_points_standard_r1': 103.0,
            'pivot_points_standard_s1': 99.0,
            'pivot_points_standard_pivot': 101.0,
            'support_resistance_20_resistance': 104.0,
            'support_resistance_20_support': 98.0,
        }
        
        bar = {'close': 105.0, 'symbol': 'TEST', 'timeframe': '5m', 'timestamp': '2024-01-01T10:00:00Z'}
        params = {'pivot_type': 'standard', 'sr_period': 20, 'breakout_threshold': 0.001}
        
        result = pivot_channel_breaks(features, bar, params)
        print(f"pivot_channel_breaks: {result is not None} - Signal: {result.get('signal_value', 'None') if result else 'None'}")
        
    except Exception as e:
        print(f"pivot_channel_breaks failed: {e}")
    
    # Test pivot_channel_bounces
    try:
        from src.strategy.strategies.indicators.structure import pivot_channel_bounces
        
        features = {
            'support_resistance_20_resistance': 104.0,
            'support_resistance_20_support': 98.0,
        }
        
        bar = {'close': 98.1, 'high': 98.5, 'low': 97.9, 'symbol': 'TEST', 'timeframe': '5m', 'timestamp': '2024-01-01T10:00:00Z'}
        params = {'sr_period': 20, 'min_touches': 2, 'bounce_threshold': 0.002}
        
        result = pivot_channel_bounces(features, bar, params)
        print(f"pivot_channel_bounces: {result is not None} - Signal: {result.get('signal_value', 'None') if result else 'None'}")
        
    except Exception as e:
        print(f"pivot_channel_bounces failed: {e}")
    
    # Test trendline_breaks
    try:
        from src.strategy.strategies.indicators.structure import trendline_breaks
        
        features = {
            'trendlines_20_2_0.002_valid_uptrends': 1,
            'trendlines_20_2_0.002_valid_downtrends': 3,
            'trendlines_20_2_0.002_nearest_support': 100.0,
            'trendlines_20_2_0.002_nearest_resistance': 105.0,
        }
        
        bar = {'close': 99.5, 'symbol': 'TEST', 'timeframe': '5m', 'timestamp': '2024-01-01T10:00:00Z'}
        params = {'pivot_lookback': 20, 'tolerance': 0.002}
        
        result = trendline_breaks(features, bar, params)
        print(f"trendline_breaks: {result is not None} - Signal: {result.get('signal_value', 'None') if result else 'None'}")
        
    except Exception as e:
        print(f"trendline_breaks failed: {e}")
    
    # Test trendline_bounces
    try:
        from src.strategy.strategies.indicators.structure import trendline_bounces
        
        features = {
            'trendlines_20_3_0.002_nearest_support': 100.0,
            'trendlines_20_3_0.002_nearest_resistance': 105.0,
            'trendlines_20_3_0.002_valid_uptrends': 3,
            'trendlines_20_3_0.002_valid_downtrends': 1,
        }
        
        bar = {'close': 100.1, 'symbol': 'TEST', 'timeframe': '5m', 'timestamp': '2024-01-01T10:00:00Z'}
        params = {'pivot_lookback': 20, 'min_touches': 3, 'tolerance': 0.002}
        
        result = trendline_bounces(features, bar, params)
        print(f"trendline_bounces: {result is not None} - Signal: {result.get('signal_value', 'None') if result else 'None'}")
        
    except Exception as e:
        print(f"trendline_bounces failed: {e}")

if __name__ == "__main__":
    test_strategies()