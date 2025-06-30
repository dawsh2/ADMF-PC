# Find the actual Sharpe calculation bug
import numpy as np

print("🐛 FINDING THE SHARPE CALCULATION BUG")
print("=" * 80)

# From the results:
# Strategy 3: Return = -0.5% → -0.0%, Sharpe = -0.25 → 0.00
# This is IMPOSSIBLE if returns are truly negative

print("\nTesting Sharpe calculation:")
print("-" * 50)

# Simulate negative returns
returns = np.array([-0.001, -0.001, 0.0005, -0.001, -0.0005])
print(f"Returns: {returns}")
print(f"Mean: {returns.mean():.6f}")
print(f"Std: {returns.std():.6f}")
print(f"Mean is negative? {returns.mean() < 0}")

# Calculate Sharpe
if returns.std() > 0:
    sharpe = returns.mean() / returns.std()
    print(f"Sharpe (no annualization): {sharpe:.3f}")
    print(f"Sharpe is negative? {sharpe < 0}")
    
    # With annualization
    annualization = np.sqrt(252 * 5 / 10)  # 5 trades over 10 days
    sharpe_ann = sharpe * annualization
    print(f"Sharpe (annualized): {sharpe_ann:.3f}")

print("\n🔍 POSSIBLE BUGS:")
print("1. Maybe returns array is empty?")
print("2. Maybe there's a max(0, sharpe) somewhere?")
print("3. Maybe the -0.0% is actually a tiny positive number?")
print("4. Maybe there's an abs() somewhere?")

# Test case that would produce Sharpe = 0.00
print("\n📊 What would give Sharpe = 0.00?")
zero_returns = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
print(f"All zeros: mean={zero_returns.mean()}, std={zero_returns.std()}")
if zero_returns.std() > 0:
    print(f"Sharpe: {zero_returns.mean() / zero_returns.std()}")
else:
    print("Sharpe: undefined (division by zero)")

# Test perfectly balanced
balanced = np.array([0.001, -0.001, 0.001, -0.001])
print(f"\nPerfectly balanced: mean={balanced.mean():.10f}")
print(f"Is mean exactly 0? {balanced.mean() == 0}")
print(f"Sharpe: {balanced.mean() / balanced.std():.10f}")

print("\n💡 INSIGHT:")
print("The -0.0% return might be a formatting issue!")
print("It could be -0.0001% which rounds to -0.0%")
print("But the Sharpe calculation might be using a different value")

# Check what -0.0% means
tiny_negative = -0.00001
print(f"\nTiny negative: {tiny_negative}")
print(f"Formatted as %: {tiny_negative*100:.1f}%")  # This would show as -0.0%!
print(f"But Sharpe with this: {tiny_negative / 0.001:.3f}")  # Still negative!

print("\n❌ CONCLUSION:")
print("There MUST be a bug in the optimization code.")
print("Possibilities:")
print("1. The code is setting Sharpe to 0 when returns.mean() is very small")
print("2. There's a max(0, sharpe) or abs(sharpe) somewhere")
print("3. The apply_stop_target function returns wrong values")
print("4. The optimization is comparing/storing values incorrectly")