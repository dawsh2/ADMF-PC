#!/usr/bin/env python3

"""Simple test of trendline strategy function loading."""

import sys
sys.path.append('/Users/daws/ADMF-PC/src')

def test_trendline_strategy_functions():
    """Test that we can load and call the trendline strategies."""
    
    try:
        # Import the strategies directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("structure", "/Users/daws/ADMF-PC/src/strategy/strategies/indicators/structure.py")
        structure_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(structure_module)
        
        # Get the strategy functions
        trendline_breaks = structure_module.trendline_breaks
        trendline_bounces = structure_module.trendline_bounces
        
        print("✓ Successfully loaded trendline strategy functions")
        
        # Test with minimal mock data
        mock_features = {
            'trendlines_20_2_0.002_valid_uptrends': 1,
            'trendlines_20_2_0.002_valid_downtrends': 2,
            'trendlines_20_2_0.002_nearest_support': 100.0,
            'trendlines_20_2_0.002_nearest_resistance': 105.0,
            'trendlines_20_3_0.002_valid_uptrends': 2,
            'trendlines_20_3_0.002_valid_downtrends': 1,
            'trendlines_20_3_0.002_nearest_support': 99.0,
            'trendlines_20_3_0.002_nearest_resistance': 106.0,
        }
        
        mock_bar = {
            'timestamp': 1000,
            'close': 102.0,
            'high': 102.5, 
            'low': 101.5,
            'symbol': 'SPY',
            'timeframe': '1m'
        }
        
        # Test trendline_breaks
        print("\nTesting trendline_breaks:")
        result1 = trendline_breaks(mock_features, mock_bar, {})
        if result1:
            print(f"✓ trendline_breaks returned: {result1['signal_value']}")
        else:
            print("✗ trendline_breaks returned None")
        
        # Test trendline_bounces  
        print("\nTesting trendline_bounces:")
        result2 = trendline_bounces(mock_features, mock_bar, {})
        if result2:
            print(f"✓ trendline_bounces returned: {result2['signal_value']}")
        else:
            print("✗ trendline_bounces returned None")
        
        return result1 is not None and result2 is not None
        
    except Exception as e:
        print(f"✗ Error testing trendline strategies: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trendline_strategy_functions()
    if success:
        print("\n🎉 Trendline strategy fixes appear to be working!")
    else:
        print("\n❌ Trendline strategies still have issues.")
    sys.exit(0 if success else 1)