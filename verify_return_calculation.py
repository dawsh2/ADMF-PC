#!/usr/bin/env python3
"""
Verify the return calculation methodology
"""
import pandas as pd
import numpy as np

def verify_returns(corrected_trades):
    """Verify different return calculation methods"""
    
    print("=== Return Calculation Verification ===\n")
    
    # Show the actual mean
    mean_return = corrected_trades['return_pct'].mean()
    print(f"Arithmetic mean return per trade: {mean_return:.4f}%")
    print(f"This means on average, each trade returns {mean_return:.4f}%")
    print()
    
    # Convert to decimal for calculations
    returns_decimal = corrected_trades['return_pct'] / 100
    
    # Method 1: Simple multiplication (your calculation)
    print("Method 1: Your calculation (1 * 1.00031^1033)")
    avg_return_decimal = mean_return / 100
    your_calc = (1 + avg_return_decimal) ** len(corrected_trades)
    print(f"Result: {your_calc:.6f} ({(your_calc-1)*100:.2f}% return)")
    print("This assumes reinvesting with the AVERAGE return on EVERY trade")
    print()
    
    # Method 2: Actual compounding (correct method)
    print("Method 2: Actual sequential compounding")
    actual_compound = (1 + returns_decimal).prod()
    print(f"Result: {actual_compound:.6f} ({(actual_compound-1)*100:.2f}% return)")
    print("This uses the ACTUAL return sequence")
    print()
    
    # Method 3: Simple sum (approximation for small returns)
    print("Method 3: Simple sum of returns")
    simple_sum = returns_decimal.sum()
    print(f"Result: {simple_sum:.6f} ({simple_sum*100:.2f}% return)")
    print("This assumes no reinvestment")
    print()
    
    # Why the difference?
    print("=== Why Your Calculation Shows 40% ===")
    print(f"You're compounding the AVERAGE return ({mean_return:.4f}%) for {len(corrected_trades)} trades")
    print("But the actual returns vary significantly:")
    print(f"  - Stop losses: -0.075%")
    print(f"  - Take profits: +0.100%") 
    print(f"  - Signal exits: vary widely")
    print()
    
    # Show distribution
    print("=== Return Distribution ===")
    print(f"Negative returns: {(corrected_trades['return_pct'] < 0).sum()} trades")
    print(f"Positive returns: {(corrected_trades['return_pct'] > 0).sum()} trades")
    print(f"Zero returns: {(corrected_trades['return_pct'] == 0).sum()} trades")
    print()
    
    # The key insight
    print("=== The Key Point ===")
    print("Your calculation assumes every trade returns the average (0.0031%)")
    print("But in reality:")
    print(f"  - {(corrected_trades['return_pct'] < 0).sum()} trades LOSE money")
    print(f"  - The losses offset much of the gains")
    print(f"  - Actual compound return: {(actual_compound-1)*100:.2f}%")
    
    # Show a few examples
    print("\n=== First 10 Trade Returns ===")
    for i in range(min(10, len(corrected_trades))):
        ret = corrected_trades.iloc[i]['return_pct']
        print(f"Trade {i+1}: {ret:+.4f}%")

# If you have corrected_trades, run:
# verify_returns(corrected_trades)

print("Run this with: verify_returns(corrected)")