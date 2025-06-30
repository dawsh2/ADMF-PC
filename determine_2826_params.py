#!/usr/bin/env python3
"""
Determine exact parameters for strategy 3 (2826 signals).
"""

def determine_params():
    print("=== DETERMINING PARAMETERS FOR STRATEGY 3 ===\n")
    
    # Parameter grid from config
    periods = [10, 15, 20, 30, 50]
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    # With granularity 25, strategies are generated in order:
    # For each filter type, for each multiplier, for each period
    
    print("Parameter mapping:")
    print("-" * 60)
    
    strategy_num = 0
    for mult_idx, mult in enumerate(multipliers):
        for period_idx, period in enumerate(periods):
            if strategy_num == 3:
                print(f"Strategy {strategy_num}: period={period}, multiplier={mult} ← FOUND!")
                return period, mult
            else:
                print(f"Strategy {strategy_num}: period={period}, multiplier={mult}")
            strategy_num += 1
            
            if strategy_num > 5:  # Just show first few
                print("...")
                break
        if strategy_num > 5:
            break
    
    # Based on position 3
    print("\nStrategy 3 parameters:")
    print("Period: 30")
    print("Multiplier: 1.0")

if __name__ == "__main__":
    determine_params()