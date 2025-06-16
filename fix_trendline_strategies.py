#!/usr/bin/env python3

"""Fix the trendline strategies by understanding the feature key mismatch."""

import sys
sys.path.append('/Users/daws/ADMF-PC/src')

def analyze_trendline_strategy_params():
    """Analyze the parameter mapping issues in trendline strategies."""
    
    print("Analyzing trendline strategy parameter mappings...")
    
    # From the structure.py file, let's see what the strategies expect:
    
    print("\n1. trendline_breaks strategy:")
    print("   feature_config: ['trendlines']")
    print("   param_feature_mapping:")
    print("     'pivot_lookback': 'trendlines_{pivot_lookback}'")
    print("     'tolerance': 'trendlines_{pivot_lookback}_{min_touches}_{tolerance}'")
    print("")
    print("   Default params: pivot_lookback=20, tolerance=0.002")
    print("   Missing min_touches param! This breaks the feature key generation.")
    print("   Current key attempt: 'trendlines_20_2_0.002' (hardcoded min_touches=2)")
    
    print("\n2. trendline_bounces strategy:")
    print("   feature_config: ['trendlines']") 
    print("   param_feature_mapping:")
    print("     'pivot_lookback': 'trendlines_{pivot_lookback}_{min_touches}_{tolerance}'")
    print("     'min_touches': 'trendlines_{pivot_lookback}_{min_touches}_{tolerance}'")
    print("     'tolerance': 'trendlines_{pivot_lookback}_{min_touches}_{tolerance}'")
    print("")
    print("   Default params: pivot_lookback=20, min_touches=3, tolerance=0.002")
    print("   Expected key: 'trendlines_20_3_0.002'")
    
    print("\n" + "="*60)
    print("PROBLEMS IDENTIFIED:")
    print("1. trendline_breaks has inconsistent param_feature_mapping")
    print("2. Both strategies assume specific TrendLines parameters")
    print("3. The TrendLines feature in structure.py may not produce values")
    
    print("\n" + "="*60)
    print("PROPOSED FIXES:")
    
    print("\n1. Fix trendline_breaks param_feature_mapping:")
    print("   Change from:")
    print("     'pivot_lookback': 'trendlines_{pivot_lookback}'")
    print("     'tolerance': 'trendlines_{pivot_lookback}_{min_touches}_{tolerance}'")
    print("   To:")
    print("     'pivot_lookback': 'trendlines_{pivot_lookback}_{min_touches}_{tolerance}'")
    print("     'min_touches': 'trendlines_{pivot_lookback}_{min_touches}_{tolerance}' # Add missing param")
    print("     'tolerance': 'trendlines_{pivot_lookback}_{min_touches}_{tolerance}'")
    
    print("\n2. Add missing min_touches parameter default:")
    print("   In trendline_breaks function: min_touches = params.get('min_touches', 2)")
    
    print("\n3. Fix feature key construction in both strategies:")
    print("   Use consistent key format: trendlines_{pivot_lookback}_{min_touches}_{tolerance}")

if __name__ == "__main__":
    analyze_trendline_strategy_params()