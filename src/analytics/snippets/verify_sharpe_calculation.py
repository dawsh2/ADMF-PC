# Verify the exact Sharpe calculation issue
import numpy as np

print("🔍 VERIFYING SHARPE CALCULATION")
print("=" * 80)

# Let's trace through exactly what the optimization code does
print("The optimization code calculates:")
print("sharpe = returns.mean() / returns.std() * np.sqrt(252 * len(returns) / max(1, days))")

# Example scenario from your results
# Strategy 3: Return = -0.5% → -0.0%, Sharpe = -0.25 → 0.00

# Simulate what might be happening
print("\n📊 Simulating the optimization:")

# Original strategy returns (losing)
original_returns = np.array([-0.002, -0.003, 0.001, -0.004, -0.001, 0.002, -0.003])
print(f"\nOriginal returns: {original_returns}")
print(f"Mean: {original_returns.mean():.6f}")
print(f"Std: {original_returns.std():.6f}")
print(f"Compound return: {((1 + original_returns).prod() - 1)*100:.2f}%")

# Calculate Sharpe (assuming 7 trades over 5 days)
days = 5
annualization = np.sqrt(252 * len(original_returns) / days)
original_sharpe = original_returns.mean() / original_returns.std() * annualization
print(f"Sharpe: {original_sharpe:.3f}")
print(f"Sharpe (2 decimals): {original_sharpe:.2f}")

# With stops/targets - creates more balanced returns
stopped_returns = np.array([-0.00075, -0.00075, 0.001, -0.00075, -0.00075, 0.001, -0.00075])
print(f"\nWith 0.075% stop, 0.1% target:")
print(f"Returns: {stopped_returns}")
print(f"Mean: {stopped_returns.mean():.6f}")
print(f"Std: {stopped_returns.std():.6f}")
print(f"Compound return: {((1 + stopped_returns).prod() - 1)*100:.2f}%")

# Calculate Sharpe
stopped_sharpe = stopped_returns.mean() / stopped_returns.std() * annualization
print(f"Sharpe: {stopped_sharpe:.6f}")
print(f"Sharpe (2 decimals): {stopped_sharpe:.2f}")

print("\n❓ KEY QUESTIONS:")
print(f"1. Is the mean truly negative? {stopped_returns.mean() < 0}")
print(f"2. Is the Sharpe truly negative? {stopped_sharpe < 0}")
print(f"3. Does it round to 0.00? {abs(stopped_sharpe) < 0.005}")

# Test edge case
perfectly_balanced = np.array([0.001, -0.001] * 50)  # 100 trades
print(f"\nPerfectly balanced (100 trades):")
print(f"Mean: {perfectly_balanced.mean():.10f}")
print(f"Compound return: {((1 + perfectly_balanced).prod() - 1)*100:.4f}%")
balanced_sharpe = perfectly_balanced.mean() / perfectly_balanced.std() * np.sqrt(252 * 100 / 50)
print(f"Sharpe: {balanced_sharpe:.10f}")
print(f"Sharpe (2 decimals): {balanced_sharpe:.2f}")

print("\n💡 INSIGHT:")
print("Even with perfectly balanced trades, we get:")
print("- Mean = 0.0000000000 (truly zero)")
print("- Sharpe = 0.0000000000 (truly zero)")
print("- Compound return = -0.0050% (negative due to volatility drag)")

print("\n🐛 CONCLUSION:")
print("The Sharpe CAN be exactly 0.00 if stops/targets create perfectly balanced trades")
print("But the compound return will still be negative due to volatility drag")
print("This explains the paradox in your results!")

# But wait - let's check if 5 losses at 0.075% and 4 wins at 0.1% gives mean=0
print("\n🔢 Checking exact stop/target math:")
n_stops = 5
n_targets = 4
stop_loss = -0.00075
target_win = 0.001
mean_return = (n_stops * stop_loss + n_targets * target_win) / (n_stops + n_targets)
print(f"{n_stops} stops at {stop_loss*100:.3f}%, {n_targets} targets at {target_win*100:.3f}%")
print(f"Mean return: {mean_return:.6f} ({mean_return*100:.4f}%)")
print(f"Is this positive? {mean_return > 0}")
print("So the Sharpe should be slightly positive, not zero!")

print("\n✅ FINAL ANSWER:")
print("The results showing Sharpe = 0.00 are due to rounding to 2 decimal places")
print("The actual Sharpe might be 0.0001 or -0.0001, which rounds to 0.00")
print("But you're right - if returns are truly negative, Sharpe should be negative too")