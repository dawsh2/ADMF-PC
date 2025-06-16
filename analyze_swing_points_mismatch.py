#!/usr/bin/env python3
"""
Analyze swing_points feature key mismatch between what strategies expect vs what FeatureHub provides.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def analyze_swing_points_mismatch():
    """Analyze the swing_points feature key mismatch."""
    
    print("=" * 80)
    print("SWING POINTS FEATURE KEY MISMATCH ANALYSIS")
    print("=" * 80)
    
    # What the price_action_swing strategy expects
    print("\n1. WHAT STRATEGY EXPECTS:")
    print("   Strategy: price_action_swing in structure.py")
    print("   Expected feature keys:")
    expected_keys = [
        "swing_points_high_10",  # swing_high with period parameter
        "swing_points_low_10",   # swing_low with period parameter  
        "swing_points_prev_high_10",  # prev_swing_high with period parameter
        "swing_points_prev_low_10"    # prev_swing_low with period parameter
    ]
    for key in expected_keys:
        print(f"     - {key}")
    
    print("   Code from strategy:")
    print("     swing_high = features.get(f'swing_points_high_{swing_period}')")
    print("     swing_low = features.get(f'swing_points_low_{swing_period}')")
    print("     prev_swing_high = features.get(f'swing_points_prev_high_{swing_period}')")
    print("     prev_swing_low = features.get(f'swing_points_prev_low_{swing_period}')")
    
    # What the SwingPoints feature actually provides
    print("\n2. WHAT FEATUREHUB PROVIDES:")
    print("   SwingPoints feature implementation returns:")
    actual_keys = [
        "swing_points_5_swing_high",  # Based on f"{name}_{sub_name}" pattern
        "swing_points_5_swing_low"    # Based on f"{name}_{sub_name}" pattern
    ]
    for key in actual_keys:
        print(f"     - {key}")
    
    print("   From SwingPoints.update() method:")
    print("     self._state.set_value({")
    print("         'swing_high': self._last_swing_high,")
    print("         'swing_low': self._last_swing_low")
    print("     })")
    print("   ")
    print("   FeatureHub decomposition (line 145-146 in hub.py):")
    print("     for sub_name, sub_value in value.items():")
    print("         results[f'{name}_{sub_name}'] = sub_value")
    print("   ")
    print("   So 'swing_points_5' + '_swing_high' = 'swing_points_5_swing_high'")
    
    # The mismatch
    print("\n3. THE KEY MISMATCH:")
    print("   Expected:    swing_points_high_10")
    print("   Actual:      swing_points_10_swing_high")
    print("   ")
    print("   The parameter (period) and sub-feature names are in wrong order!")
    print("   Expected:    {feature}_{subkey}_{period}")
    print("   Actual:      {feature}_{period}_{subkey}")
    
    # Missing features
    print("\n4. MISSING FEATURES:")
    print("   The SwingPoints implementation only tracks current swing points:")
    print("     - swing_high (current)")
    print("     - swing_low (current)")
    print("   ")
    print("   But the strategy also needs previous swing points:")
    print("     - prev_swing_high (MISSING)")
    print("     - prev_swing_low (MISSING)")
    print("   ")
    print("   The SwingPoints class needs to track historical swing points!")
    
    # Other strategies using swing_points
    print("\n5. OTHER STRATEGIES USING SWING_POINTS:")
    print("   Only price_action_swing uses swing_points features")
    print("   So fixing this impacts just one strategy")
    
    # Solution options
    print("\n6. SOLUTION OPTIONS:")
    print("   Option A: Fix SwingPoints to match expected key format")
    print("     - Change decomposition to put period at end")
    print("     - Add prev_swing_high and prev_swing_low tracking")
    print("   ")
    print("   Option B: Update strategy to match actual key format")  
    print("     - Change strategy to use swing_points_10_swing_high format")
    print("     - Handle missing prev_swing features")
    print("   ")
    print("   Option C: Create mapping in FeatureHub for backward compatibility")
    print("     - Add key aliases for swing_points features")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION: Option A - Fix SwingPoints implementation")
    print("This matches the pattern used by other decomposed features")
    print("=" * 80)

if __name__ == "__main__":
    analyze_swing_points_mismatch()