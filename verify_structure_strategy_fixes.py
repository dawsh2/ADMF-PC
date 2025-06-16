#!/usr/bin/env python3

"""
Verify that the structure strategy fixes work correctly.
"""

import sys
sys.path.insert(0, 'src')

def test_all_fixed_strategies():
    """Test all the fixed structure strategies."""
    print("=" * 60)
    print("TESTING FIXED STRUCTURE STRATEGIES")
    print("=" * 60)
    
    results = {}
    
    # Import the strategy functions
    try:
        from src.strategy.strategies.indicators.structure import (
            pivot_channel_breaks,
            pivot_channel_bounces, 
            trendline_breaks,
            trendline_bounces
        )
        print("✓ All strategy imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return
    
    # Test 1: pivot_channel_breaks
    print("\n1. Testing pivot_channel_breaks...")
    try:
        features = {
            'pivot_points_standard_r1': 103.0,
            'pivot_points_standard_s1': 99.0,
            'pivot_points_standard_pivot': 101.0,
            'support_resistance_20_resistance': 104.0,
            'support_resistance_20_support': 98.0,
        }
        
        # Test breakout above resistance
        bar_breakout = {
            'close': 105.0,  # Above R1 and resistance
            'symbol': 'SPY',
            'timeframe': '5m',
            'timestamp': '2024-01-01T10:00:00Z'
        }
        
        params = {
            'pivot_type': 'standard',
            'sr_period': 20,
            'breakout_threshold': 0.001
        }
        
        result = pivot_channel_breaks(features, bar_breakout, params)
        if result and result.get('signal_value') == 1:
            print("  ✓ Bullish breakout signal generated correctly")
            results['pivot_channel_breaks'] = 'FIXED'
        else:
            print(f"  ✗ Expected bullish signal, got: {result}")
            results['pivot_channel_breaks'] = 'FAILED'
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['pivot_channel_breaks'] = 'ERROR'
    
    # Test 2: pivot_channel_bounces  
    print("\n2. Testing pivot_channel_bounces...")
    try:
        features = {
            'support_resistance_20_resistance': 104.0,
            'support_resistance_20_support': 98.0,
        }
        
        # Test bounce from support
        bar_bounce = {
            'close': 98.1,   # Just above support 
            'high': 98.5,
            'low': 97.8,     # Touched support
            'symbol': 'SPY',
            'timeframe': '5m',
            'timestamp': '2024-01-01T10:00:00Z'
        }
        
        params = {
            'sr_period': 20,
            'min_touches': 2,
            'bounce_threshold': 0.002
        }
        
        result = pivot_channel_bounces(features, bar_bounce, params)
        if result and result.get('signal_value') == 1:
            print("  ✓ Support bounce signal generated correctly")
            results['pivot_channel_bounces'] = 'FIXED'
        else:
            print(f"  ✗ Expected bounce signal, got: {result}")
            results['pivot_channel_bounces'] = 'FAILED'
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['pivot_channel_bounces'] = 'ERROR'
    
    # Test 3: trendline_breaks
    print("\n3. Testing trendline_breaks...")
    try:
        features = {
            'trendlines_20_2_0.002_valid_uptrends': 3,  # More uptrends than downtrends
            'trendlines_20_2_0.002_valid_downtrends': 1,
            'trendlines_20_2_0.002_nearest_support': 100.0,
            'trendlines_20_2_0.002_nearest_resistance': 105.0,
        }
        
        # Test support break
        bar_break = {
            'close': 99.5,  # Below support 
            'symbol': 'SPY',
            'timeframe': '5m',
            'timestamp': '2024-01-01T10:00:00Z'
        }
        
        params = {
            'pivot_lookback': 20,
            'tolerance': 0.002
        }
        
        result = trendline_breaks(features, bar_break, params)
        if result and result.get('signal_value') == -1:
            print("  ✓ Trendline break signal generated correctly")
            results['trendline_breaks'] = 'FIXED'
        else:
            print(f"  ✗ Expected bearish break signal, got: {result}")
            results['trendline_breaks'] = 'FAILED'
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['trendline_breaks'] = 'ERROR'
    
    # Test 4: trendline_bounces
    print("\n4. Testing trendline_bounces...")
    try:
        features = {
            'trendlines_20_3_0.002_nearest_support': 100.0,
            'trendlines_20_3_0.002_nearest_resistance': 105.0,
            'trendlines_20_3_0.002_valid_uptrends': 3,
            'trendlines_20_3_0.002_valid_downtrends': 1,
        }
        
        # Test bounce from support trendline
        bar_bounce = {
            'close': 100.1,  # Very close to support
            'symbol': 'SPY',
            'timeframe': '5m',
            'timestamp': '2024-01-01T10:00:00Z'
        }
        
        params = {
            'pivot_lookback': 20,
            'min_touches': 3,
            'tolerance': 0.002
        }
        
        result = trendline_bounces(features, bar_bounce, params)
        if result and result.get('signal_value') == 1:
            print("  ✓ Trendline bounce signal generated correctly")
            results['trendline_bounces'] = 'FIXED'
        else:
            print(f"  ✗ Expected bounce signal, got: {result}")
            results['trendline_bounces'] = 'FAILED'
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['trendline_bounces'] = 'ERROR'
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF STRUCTURE STRATEGY FIXES")
    print("=" * 60)
    
    for strategy, status in results.items():
        status_symbol = "✓" if status == "FIXED" else "✗"
        print(f"{status_symbol} {strategy}: {status}")
    
    fixed_count = sum(1 for status in results.values() if status == 'FIXED')
    total_count = len(results)
    
    print(f"\nFixed strategies: {fixed_count}/{total_count}")
    
    if fixed_count == total_count:
        print("\n🎉 ALL STRUCTURE STRATEGIES SUCCESSFULLY FIXED!")
        print("\nThe key mismatches between strategies and FeatureHub decomposition")
        print("have been resolved. These strategies should now generate signals")
        print("when run in the actual grid search.")
    else:
        print(f"\n⚠️  {total_count - fixed_count} strategies still need attention")

if __name__ == "__main__":
    test_all_fixed_strategies()