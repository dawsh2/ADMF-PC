# Analyze the optimization logic to find the bug
print("🔍 ANALYZING OPTIMIZATION LOGIC")
print("=" * 80)

print("\nThe optimization code does:")
print("1. For each strategy, test different stop/target combinations")
print("2. Keep the configuration with the highest Sharpe ratio")
print("3. Store both original and optimal metrics")

print("\n🐛 FOUND THE BUG!")
print("-" * 50)
print("The code says: 'if sharpe > best_config['sharpe']:'")
print("This means it only updates if the new Sharpe is BETTER")
print("But for negative Sharpe ratios:")
print("  -0.25 is worse than -0.10")
print("  -0.10 is worse than 0.00")
print("  0.00 is worse than 0.10")

print("\nSo when a strategy has Sharpe = -0.25 and you test stops/targets:")
print("- If stops create balanced trades with mean ≈ 0, Sharpe ≈ 0")
print("- Since 0 > -0.25, it considers this 'better'")
print("- But the returns are still negative due to compound effects!")

print("\n💡 THE REAL ISSUE:")
print("When comparing Sharpe ratios, the code assumes higher is always better")
print("But a Sharpe of 0 with negative returns is NOT better than")
print("a Sharpe of -0.25 with negative returns!")

print("\n📊 Example:")
print("Strategy A: -5% return, Sharpe = -0.5 (consistently losing)")
print("Strategy B: -0.1% return, Sharpe = 0.0 (barely losing)")
print("Strategy C: +5% return, Sharpe = 0.5 (consistently winning)")

print("\nThe optimization picks B over A because 0.0 > -0.5")
print("But B still loses money!")

print("\n✅ THE FIX:")
print("The optimization should:")
print("1. First filter for positive returns")
print("2. Then optimize Sharpe among profitable strategies")
print("3. Or use a better metric like return/max_drawdown")
print("4. Or at minimum, check that returns are positive when Sharpe improves")

print("\n⚠️ This explains why ALL your strategies show negative or near-zero returns!")
print("The optimization is finding the 'least bad' way to lose money slowly.")