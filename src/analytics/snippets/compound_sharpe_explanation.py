# Explaining Compound Sharpe Ratio (using geometric mean)
import numpy as np

print("📊 COMPOUND SHARPE RATIO EXPLANATION")
print("=" * 80)

print("\n1. TRADITIONAL SHARPE (Arithmetic)")
print("-" * 40)
print("Uses arithmetic mean: sum(returns) / n")
print("Sharpe = mean(returns) / std(returns) * sqrt(periods_per_year)")

print("\n2. COMPOUND SHARPE (Geometric)")
print("-" * 40)
print("Uses geometric mean: (product(1 + returns))^(1/n) - 1")
print("Sharpe = geometric_mean / std(returns) * sqrt(periods_per_year)")

# Example to show the difference
returns = np.array([0.10, -0.08, 0.06, -0.04, 0.05, -0.03])
print(f"\nExample returns: {returns}")

# Arithmetic mean and Sharpe
arith_mean = returns.mean()
arith_sharpe = arith_mean / returns.std() * np.sqrt(252/6)
print(f"\nArithmetic mean: {arith_mean:.4f} ({arith_mean*100:.2f}%)")
print(f"Arithmetic Sharpe: {arith_sharpe:.3f}")

# Geometric mean and Sharpe
compound_return = (1 + returns).prod() - 1
geom_mean = (1 + compound_return)**(1/len(returns)) - 1
geom_sharpe = geom_mean / returns.std() * np.sqrt(252/6)
print(f"\nCompound return: {compound_return:.4f} ({compound_return*100:.2f}%)")
print(f"Geometric mean: {geom_mean:.4f} ({geom_mean*100:.2f}%)")
print(f"Geometric Sharpe: {geom_sharpe:.3f}")

print("\n💡 KEY INSIGHT:")
print("Geometric mean is ALWAYS ≤ arithmetic mean (due to volatility drag)")
print("For volatile returns, the difference can be significant")
print("Compound Sharpe better reflects what investors actually experience")

# Show how this affects optimization
print("\n🎯 WHY THIS MATTERS FOR OPTIMIZATION:")
print("-" * 50)

# Strategy A: Low volatility, slight negative drift
strat_a = np.array([-0.001] * 10)
# Strategy B: Higher volatility, same arithmetic mean
strat_b = np.array([0.009, -0.011, 0.010, -0.012, 0.008, -0.010, 0.009, -0.011, 0.010, -0.012])

print(f"\nStrategy A (consistent small losses): {strat_a}")
print(f"Arithmetic mean: {strat_a.mean():.4f}")
print(f"Compound return: {((1 + strat_a).prod() - 1)*100:.2f}%")
print(f"Traditional Sharpe: {strat_a.mean() / strat_a.std() if strat_a.std() > 0 else 'undefined'}")

print(f"\nStrategy B (volatile, same mean): {strat_b}")
print(f"Arithmetic mean: {strat_b.mean():.4f}")
print(f"Compound return: {((1 + strat_b).prod() - 1)*100:.2f}%")
print(f"Traditional Sharpe: {strat_b.mean() / strat_b.std():.3f}")

print("\n❌ Traditional Sharpe picks Strategy A (lower volatility)")
print("✅ But Strategy B has better compound returns!")
print("\nCompound Sharpe would correctly identify this.")