# Diagnose the impossible positive Sharpe with negative returns issue
import pandas as pd
import numpy as np
from pathlib import Path

print("🔍 DIAGNOSING SHARPE RATIO BUG")
print("=" * 80)

# The issue: Strategies showing negative returns but positive Sharpe ratios
# This is mathematically impossible!

print("\n❌ IMPOSSIBLE RESULTS DETECTED:")
print("Strategy 1: Return = 0.6% → 0.5%, Sharpe = 0.51 → 0.62")
print("  - Returns are positive, Sharpe is positive ✓ Makes sense")
print("\nStrategy 3: Return = -0.5% → -0.0%, Sharpe = -0.25 → 0.00")
print("  - Return is negative (-0.0%), but Sharpe is 0.00 ❌ IMPOSSIBLE!")

print("\n💡 HYPOTHESIS 1: Sharpe calculation error")
print("If using trade returns, the formula should be:")
print("  mean(trade_returns) / std(trade_returns) * sqrt(annualization_factor)")
print("If mean is negative, Sharpe MUST be negative")

print("\n💡 HYPOTHESIS 2: Return calculation error")
print("Maybe the displayed 'Total Return' is calculated differently than expected")
print("  - Could be using sum instead of compound?")
print("  - Could be a display formatting issue?")

print("\n💡 HYPOTHESIS 3: Sign error in optimization")
print("The optimization might be flipping signs somewhere")

print("\n📊 Let's manually check the math:")
print("\nExample: If a strategy has these trade returns:")
trade_returns = np.array([-0.001, -0.001, 0.001, -0.001, -0.001])  # Mostly losing
print(f"Trade returns: {trade_returns}")
print(f"Mean return: {trade_returns.mean():.6f}")
print(f"Std deviation: {trade_returns.std():.6f}")
print(f"Sharpe (no annualization): {trade_returns.mean() / trade_returns.std():.3f}")
print(f"Total return (sum): {trade_returns.sum():.6f}")
print(f"Total return (compound): {((1 + trade_returns).prod() - 1):.6f}")

print("\n🐛 BUG FOUND:")
print("If Sharpe = 0.00 with negative returns, it means:")
print("1. The numerator (mean return) is being set to 0 somehow")
print("2. OR there's a max(0, sharpe) somewhere in the code")
print("3. OR the calculation is using abs() somewhere it shouldn't")

print("\n🔧 TO FIX:")
print("1. Check the apply_stop_target function's Sharpe calculation")
print("2. Ensure no max(0, sharpe) or abs() operations") 
print("3. Verify return calculations match between display and Sharpe calc")