#!/usr/bin/env python3

"""Test the fixed trendline strategies."""

import sys
sys.path.append('/Users/daws/ADMF-PC/src')

from strategy.strategies.indicators.structure import trendline_breaks, trendline_bounces

def test_trendline_strategies():
    """Test the trendline strategies with mock features."""
    
    # Mock features with the expected decomposed keys
    mock_features = {
        # For trendline_breaks with default params (pivot_lookback=20, min_touches=2, tolerance=0.002)
        'trendlines_20_2_0.002_valid_uptrends': 3,
        'trendlines_20_2_0.002_valid_downtrends': 1,
        'trendlines_20_2_0.002_nearest_support': 100.0,
        'trendlines_20_2_0.002_nearest_resistance': 105.0,
        
        # For trendline_bounces with default params (pivot_lookback=20, min_touches=3, tolerance=0.002)
        'trendlines_20_3_0.002_valid_uptrends': 2,
        'trendlines_20_3_0.002_valid_downtrends': 4,
        'trendlines_20_3_0.002_nearest_support': 99.5,
        'trendlines_20_3_0.002_nearest_resistance': 104.5,
    }
    
    mock_bar = {
        'timestamp': 1000,
        'close': 102.0,
        'high': 102.5,
        'low': 101.5,
        'symbol': 'SPY',
        'timeframe': '1m'
    }
    
    print("Testing trendline_breaks strategy:")
    print("=" * 40)
    
    # Test with default params
    params_1 = {}
    result_1 = trendline_breaks(mock_features, mock_bar, params_1)
    print(f"Default params: {result_1}")
    
    # Test with custom params
    params_2 = {'pivot_lookback': 20, 'min_touches': 2, 'tolerance': 0.002}
    result_2 = trendline_breaks(mock_features, mock_bar, params_2)
    print(f"Custom params: {result_2}")
    
    print("\nTesting trendline_bounces strategy:")
    print("=" * 40)
    
    # Test with default params
    params_3 = {}
    result_3 = trendline_bounces(mock_features, mock_bar, params_3)
    print(f"Default params: {result_3}")
    
    # Test with custom params
    params_4 = {'pivot_lookback': 20, 'min_touches': 3, 'tolerance': 0.002}
    result_4 = trendline_bounces(mock_features, mock_bar, params_4)
    print(f"Custom params: {result_4}")
    
    print("\nFeature key analysis:")
    print("=" * 40)
    
    # Show which keys each strategy looks for
    print("trendline_breaks looks for:")
    for key in mock_features:
        if '20_2_0.002' in key:
            print(f"  {key}: {mock_features[key]}")
    
    print("\ntrendline_bounces looks for:")
    for key in mock_features:
        if '20_3_0.002' in key:
            print(f"  {key}: {mock_features[key]}")
    
    # Test success criteria
    print("\nTest Results:")
    print("=" * 40)
    success_count = 0
    total_tests = 4
    
    if result_1 is not None:
        print("✓ trendline_breaks with default params: SUCCESS")
        success_count += 1
    else:
        print("✗ trendline_breaks with default params: FAILED")
    
    if result_2 is not None:
        print("✓ trendline_breaks with custom params: SUCCESS")
        success_count += 1
    else:
        print("✗ trendline_breaks with custom params: FAILED")
    
    if result_3 is not None:
        print("✓ trendline_bounces with default params: SUCCESS")
        success_count += 1
    else:
        print("✗ trendline_bounces with default params: FAILED")
    
    if result_4 is not None:
        print("✓ trendline_bounces with custom params: SUCCESS")
        success_count += 1
    else:
        print("✗ trendline_bounces with custom params: FAILED")
    
    print(f"\nOverall: {success_count}/{total_tests} tests passed")
    return success_count == total_tests

if __name__ == "__main__":
    success = test_trendline_strategies()
    if success:
        print("\n🎉 All trendline strategy fixes are working!")
    else:
        print("\n❌ Some trendline strategies still have issues.")
    sys.exit(0 if success else 1)