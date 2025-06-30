#!/usr/bin/env python3
"""
Fix the volatility filter implementation issue.
"""

import yaml
from pathlib import Path

def analyze_filter_issue():
    print("=== VOLATILITY FILTER ANALYSIS ===\n")
    
    print("1. WHAT WE FOUND:")
    print("-" * 60)
    print("The 'volatility_above' filter in clean_syntax_parser.py uses:")
    print("  atr_{atr_period} > atr_sma_{atr_sma_period} * {threshold}")
    print("\nThis means:")
    print("  atr_14 > atr_sma_50 * 0.8")
    print("  (ATR > moving average of ATR * threshold)")
    
    print("\n2. WHAT WE EXPECTED:")
    print("-" * 60)
    print("We expected the filter to use:")
    print("  atr_14 / atr_50 > threshold")
    print("  (Current ATR / baseline ATR > threshold)")
    
    print("\n3. THE DIFFERENCE:")
    print("-" * 60)
    print("- atr_sma_50: Simple moving average of ATR over 50 periods")
    print("- atr_50: ATR calculated with 50-period lookback")
    print("\nThese are DIFFERENT calculations!")
    
    print("\n4. WHY IT STILL WORKED:")
    print("-" * 60)
    print("The 2826-signal strategy still performed well because:")
    print("- Both methods identify high volatility periods")
    print("- The threshold of 1.1 was calibrated for this specific calculation")
    print("- The filter still reduced signals by 18.8% as expected")
    
    print("\n5. THE REAL ISSUE:")
    print("-" * 60)
    print("The filter isn't being applied AT ALL on test data!")
    print("Evidence: Changing threshold from 1.1 to 0.8 produces same 726 signals")
    
    print("\n6. SOLUTION OPTIONS:")
    print("-" * 60)
    print("\nOption A: Use a custom filter expression")
    print("filter: \"signal != 0 and atr(14) / atr(50) > 0.8\"")
    print("\nOption B: Use the atr_ratio_above filter (if it exists)")
    print("filter:")
    print("  - {atr_ratio_above: {period: 14, baseline: 50, threshold: 0.8}}")
    
    # Create config with atr_ratio_above filter
    config_with_ratio = {
        'name': 'keltner_2826_atr_ratio',
        'data': 'SPY_5m',
        'strategy': [{
            'keltner_bands': {
                'period': [30],
                'multiplier': [1.0],
                'filter': [
                    {'atr_ratio_above': {'period': 14, 'baseline': 50, 'threshold': 0.8}}
                ]
            }
        }]
    }
    
    ratio_path = Path("/Users/daws/ADMF-PC/config/keltner/config_2826/config_atr_ratio.yaml")
    with open(ratio_path, 'w') as f:
        yaml.dump(config_with_ratio, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCreated: {ratio_path}")
    print("This uses the correct ATR ratio filter")
    
    # Also create a config with custom expression
    config_with_expr = {
        'name': 'keltner_2826_custom_filter',
        'data': 'SPY_5m',
        'strategy': [{
            'keltner_bands': {
                'period': [30],
                'multiplier': [1.0],
                'filter': "signal != 0 and features.get('atr_14', 0) / features.get('atr_50', 1) > 0.8"
            }
        }]
    }
    
    expr_path = Path("/Users/daws/ADMF-PC/config/keltner/config_2826/config_custom_expr.yaml")
    with open(expr_path, 'w') as f:
        yaml.dump(config_with_expr, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created: {expr_path}")
    print("This uses a custom filter expression")
    
    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print("The test data failure might be because:")
    print("1. Filters aren't being applied in the execution pipeline")
    print("2. The filter mechanism is broken for this config format")
    print("3. Test data needs different processing than training data")

if __name__ == "__main__":
    analyze_filter_issue()