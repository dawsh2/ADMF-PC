# Demonstrate how we can get Sharpe = 0.00 with negative compound returns
import numpy as np

print("🔬 DEMONSTRATING SHARPE = 0.00 WITH NEGATIVE RETURNS")
print("=" * 80)

# Create returns that have arithmetic mean ≈ 0 but negative compound return
# This happens when stops/targets create symmetric wins and losses
returns = np.array([
    0.001,   # Win (hit target)
    -0.00075,  # Loss (hit stop) 
    0.001,   # Win
    -0.00075,  # Loss
    0.001,   # Win
    -0.00075,  # Loss
    0.001,   # Win  
    -0.00075,  # Loss
])

print("Stop loss = 0.075%, Target = 0.1%")
print("This creates 4:3 win/loss ratio due to 1.33:1 reward/risk")
print(f"\nReturns: {returns}")

# Arithmetic calculations (used for Sharpe)
arithmetic_mean = returns.mean()
print(f"\nArithmetic mean: {arithmetic_mean:.10f}")
print(f"Is this exactly 0? {arithmetic_mean == 0}")
print(f"Is this close to 0? {abs(arithmetic_mean) < 1e-10}")

# For display purposes, this might round to 0
print(f"Formatted as percentage: {arithmetic_mean*100:.1f}%")  # Shows 0.0%

# Sharpe calculation
if returns.std() > 0:
    sharpe = arithmetic_mean / returns.std()
    print(f"\nSharpe ratio: {sharpe:.10f}")
    print(f"Sharpe rounded to 2 decimals: {sharpe:.2f}")  # Shows 0.00
    
# Compound return (what actually matters)
compound_return = (1 + returns).prod() - 1
print(f"\nCompound return: {compound_return:.6f}")
print(f"Compound return %: {compound_return*100:.1f}%")  # Shows -0.0%

print("\n💡 EXPLANATION:")
print("1. With 4 wins at +0.1% and 4 losses at -0.075%")
print("2. Arithmetic: 4*(0.001) + 4*(-0.00075) = 0.001")
print("3. This gives mean = 0.000125 ≈ 0")
print("4. Sharpe = 0.000125 / std ≈ 0.089")
print("5. Displayed as 0.00 when rounded to 2 decimals")
print("6. But compound return is negative due to volatility!")

print("\n❓ BUT WAIT:")
print("Even if mean = 0.000125, Sharpe should be positive, not 0.00")
print("Unless...")

# The actual bug might be even simpler
print("\n🐛 POSSIBLE BUG IN PANDAS/NUMPY:")
very_small = 1e-16
print(f"Very small number: {very_small}")
print(f"Divided by 0.001: {very_small / 0.001}")
print(f"Is this 0 in practice? {very_small / 0.001 == 0}")
print(f"Rounded: {round(very_small / 0.001, 2)}")

print("\n✅ CONCLUSION:")
print("The Sharpe = 0.00 could be due to:")
print("1. Rounding for display (most likely)")
print("2. Numerical precision issues") 
print("3. Or the arithmetic mean is SO close to 0 that Sharpe rounds to 0.00")
print("\nBut you're right - it should show as slightly negative or positive,")
print("not exactly 0.00, unless the mean is EXACTLY 0.")