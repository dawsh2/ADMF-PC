# Reproduce and fix the Sharpe ratio bug
import pandas as pd
import numpy as np

print("🐛 REPRODUCING SHARPE RATIO BUG")
print("=" * 80)

# Simulate what happens in the optimization
# Strategy 3: Return = -0.5% → -0.0%, Sharpe = -0.25 → 0.00

# Let's say we have some trade returns that average negative
trade_returns = np.array([0.001, -0.002, 0.001, -0.002, -0.001, -0.001, 0.001, -0.001])
print(f"Sample trade returns: {trade_returns}")
print(f"Number of trades: {len(trade_returns)}")
print(f"Mean return per trade: {trade_returns.mean():.6f}")
print(f"Std dev of returns: {trade_returns.std():.6f}")
print(f"Sum of returns: {trade_returns.sum():.6f}")
print(f"Compound return: {((1 + trade_returns).prod() - 1):.6f}")

# Calculate Sharpe the way the notebook does
days_in_sample = 10  # Assume 10 days of trading
annualization_factor = np.sqrt(252 * len(trade_returns) / max(1, days_in_sample))
print(f"\nAnnualization factor: {annualization_factor:.2f}")

if trade_returns.std() > 0:
    sharpe = trade_returns.mean() / trade_returns.std() * annualization_factor
    print(f"Calculated Sharpe: {sharpe:.3f}")
else:
    print("Standard deviation is 0!")

print("\n🔍 INVESTIGATING THE SHARPE = 0.00 CASE")
print("When does Sharpe = 0.00 exactly?")
print("1. When mean return = 0 (perfectly balanced wins/losses)")
print("2. When std dev = infinity (impossible)")
print("3. When there's a bug in the code")

# Test edge case
zero_mean_returns = np.array([0.001, -0.001, 0.001, -0.001])
print(f"\nZero mean returns: {zero_mean_returns}")
print(f"Mean: {zero_mean_returns.mean():.10f}")
print(f"Sharpe: {zero_mean_returns.mean() / zero_mean_returns.std() if zero_mean_returns.std() > 0 else 'N/A'}")

print("\n💡 LIKELY CAUSE:")
print("The optimization might be finding configurations where:")
print("1. Stops/targets create perfectly balanced wins and losses")
print("2. The mean return becomes very close to 0")
print("3. But the COMPOUND return is still negative due to volatility drag")

# Demonstrate volatility drag
balanced_returns = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
print(f"\nBalanced returns: {balanced_returns}")
print(f"Mean (arithmetic): {balanced_returns.mean():.6f}")
print(f"Total (sum): {balanced_returns.sum():.6f}")
print(f"Total (compound): {((1 + balanced_returns).prod() - 1):.6f}")
print("☝️ Note: Mean is 0, but compound return is negative!")

print("\n🐛 THE BUG:")
print("The code shows 'Total Return' which is likely the compound return")
print("But calculates Sharpe using arithmetic mean of returns")
print("When arithmetic mean ≈ 0, Sharpe ≈ 0")
print("But compound return can still be negative!")

print("\n✅ This explains why we see:")
print("- Negative total returns (compound)")
print("- Zero or positive Sharpe (arithmetic mean ≈ 0)")
print("- This is technically correct but very misleading!")