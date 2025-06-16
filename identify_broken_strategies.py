#!/usr/bin/env python3
"""
Identify which strategies need fixes based on previous analysis.
"""

import json
from pathlib import Path


def main():
    """Analyze known issues with strategies"""
    
    # Load previous analysis
    with open('strategy_analysis_results.json', 'r') as f:
        analysis = json.load(f)
    
    print("=" * 80)
    print("STRATEGY TYPES REQUIRING FIXES")
    print("=" * 80)
    
    # From the conversation, we know:
    # - Started with 14 working strategies
    # - After fixes, got to 28 working strategies
    # - Expected 36 total strategy types
    # - So 8 strategies still need fixes
    
    # Based on the analysis file, these are the strategy types that were found
    # but may not be generating signals:
    
    problem_strategies = [
        # Volume-based strategies (likely missing volume features)
        ('accumulation_distribution', 'Uses ad feature - may not be computed'),
        ('obv_trend', 'Uses obv feature - may not be computed'),
        ('vwap_deviation', 'Uses vwap feature - may not be computed'),
        
        # Complex indicators that may have implementation issues
        ('adx_trend_strength', 'ADX calculation may be incomplete'),
        ('aroon_crossover', 'Aroon indicator logic may be missing'),
        ('fibonacci_retracement', 'Fibonacci levels calculation needed'),
        ('ichimoku_cloud_position', 'Complex Ichimoku cloud logic'),
        ('parabolic_sar', 'SAR calculation may be incomplete'),
        
        # Price structure strategies
        ('pivot_points', 'Pivot point calculation logic needed'),
        ('price_action_swing', 'Swing detection logic needed'),
        ('support_resistance_breakout', 'S/R level detection needed'),
        
        # Oscillators that may need fixes
        ('stochastic_rsi', 'StochRSI calculation chain needed'),
        ('ultimate_oscillator', 'Multi-period oscillator logic'),
        ('vortex_crossover', 'Vortex indicator calculation'),
        
        # Trend/momentum indicators
        ('linear_regression_slope', 'Linear regression calculation'),
        ('roc_threshold', 'Rate of change threshold logic'),
        ('supertrend', 'Supertrend calculation logic'),
    ]
    
    print("\nStrategy types likely not generating signals:\n")
    
    for i, (strategy_type, issue) in enumerate(problem_strategies, 1):
        if strategy_type in analysis:
            features = analysis[strategy_type].get('features_needed', [])
            print(f"{i:2d}. {strategy_type}")
            print(f"    Issue: {issue}")
            if features:
                print(f"    Features needed: {', '.join(features)}")
            print()
    
    print("\n" + "=" * 80)
    print("PRIORITY FIXES:")
    print("=" * 80)
    
    print("\n1. VOLUME FEATURES (3 strategies):")
    print("   - accumulation_distribution (needs 'ad' feature)")
    print("   - obv_trend (needs 'obv' feature)")  
    print("   - vwap_deviation (needs 'vwap' feature)")
    
    print("\n2. COMPLEX INDICATORS (5 strategies):")
    print("   - ichimoku_cloud_position")
    print("   - fibonacci_retracement")
    print("   - pivot_points")
    print("   - supertrend")
    print("   - parabolic_sar")
    
    print("\n3. OSCILLATORS (3 strategies):")
    print("   - stochastic_rsi")
    print("   - ultimate_oscillator")
    print("   - vortex_crossover")
    
    print("\n4. STRUCTURE/PATTERN (3 strategies):")
    print("   - price_action_swing")
    print("   - support_resistance_breakout")
    print("   - linear_regression_slope")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED APPROACH:")
    print("=" * 80)
    
    print("\n1. First, check if volume features (ad, obv, vwap) are being computed")
    print("2. Then verify each strategy's signal generation logic")
    print("3. Focus on one category at a time for systematic fixes")
    
    # Count by category
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Total strategy types found: {len(analysis)}")
    print(f"Likely not working: ~8-14 strategies")
    print(f"Categories affected: Volume, Complex Indicators, Oscillators, Structure")


if __name__ == "__main__":
    main()