#!/usr/bin/env python3
"""
Clarify percentage vs decimal returns
"""

def clarify_percentage():
    print("=== Clarifying Percentage vs Decimal ===\n")
    
    # The mean return
    mean_return_pct = 0.0031  # This is 0.0031%
    
    print(f"Mean return from data: {mean_return_pct}")
    print(f"This represents: {mean_return_pct}%")
    print(f"As a decimal: {mean_return_pct/100} = {mean_return_pct/100:.6f}")
    print()
    
    # Compounding
    n_trades = 1033
    
    print("For 1033 trades:")
    print(f"\nCorrect calculation:")
    print(f"  (1 + {mean_return_pct/100:.6f})^{n_trades}")
    correct_result = (1 + mean_return_pct/100) ** n_trades
    print(f"  = {correct_result:.4f}")
    print(f"  = {(correct_result-1)*100:.2f}% total return")
    
    print(f"\nIncorrect calculation (treating as decimal):")
    print(f"  (1 + {mean_return_pct:.4f})^{n_trades}")
    incorrect_result = (1 + mean_return_pct) ** n_trades
    print(f"  = {incorrect_result:.4f}")
    print(f"  = {(incorrect_result-1)*100:.2f}% total return")
    
    print("\n=== The DataFrame Format ===")
    print("In your trades_df:")
    print("  return_pct = 0.1 means 0.1% (not 10%)")
    print("  return_pct = 0.075 means 0.075% (not 7.5%)")
    print("  return_pct = -0.1 means -0.1% (not -10%)")
    
    print("\nSo when we calculate mean(return_pct) = 0.0031")
    print("This is 0.0031% average return per trade")

clarify_percentage()