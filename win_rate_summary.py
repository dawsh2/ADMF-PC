#!/usr/bin/env python3
"""Compile win rates for all timeframe strategies."""

# From our previous analyses
results = {
    '5m_basic': {
        'trades': 162,
        'win_rate': 0.654,  # 65.4%
        'annual_return': 0.0259,
        'avg_win': 0.0017,
        'avg_loss': -0.0027,
        'profit_factor': 1.15
    },
    '5m_tuned': {
        'trades': 63,
        'win_rate': 0.635,  # 63.5%
        'annual_return': 0.0186,
        'avg_win': 0.0015,
        'avg_loss': -0.0019,
        'profit_factor': 1.37
    },
    '15m_basic': {
        'trades': 42,
        'win_rate': 0.619,  # 61.9%
        'annual_return': 0.0272,
        'avg_win': 0.0031,
        'avg_loss': -0.0036,
        'profit_factor': 1.40
    },
    '15m_optimized': {
        'trades': 16,
        'win_rate': 0.562,  # 56.2%
        'annual_return': -0.0038,
        'avg_win': 0.0023,
        'avg_loss': -0.0034,
        'profit_factor': 0.87
    }
}

print("=== WIN RATE SUMMARY ===")
print("All strategies with 0.5 bps execution cost\n")

print(f"{'Strategy':<15} {'Win Rate':<10} {'Trades':<8} {'Avg Win':<10} {'Avg Loss':<10} {'Win/Loss':<10} {'Annual':<10}")
print("-" * 80)

for name, data in results.items():
    win_loss_ratio = abs(data['avg_win'] / data['avg_loss']) if data['avg_loss'] != 0 else 0
    print(f"{name:<15} {data['win_rate']*100:<10.1f}% {data['trades']:<8} "
          f"{data['avg_win']*100:<10.2f}% {data['avg_loss']*100:<10.2f}% "
          f"{win_loss_ratio:<10.2f} {data['annual_return']*100:<10.2f}%")

print("\n=== KEY WIN RATE INSIGHTS ===\n")

print("1. Win Rate Ranking:")
sorted_by_wr = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)
for i, (name, data) in enumerate(sorted_by_wr):
    print(f"   {i+1}. {name}: {data['win_rate']*100:.1f}%")

print("\n2. Win Rate vs Returns:")
print("   - Highest win rate (5m_basic: 65.4%) → 2nd best returns")
print("   - Best returns (15m_basic: 2.72%) → 3rd best win rate (61.9%)")
print("   - Lowest win rate (15m_opt: 56.2%) → negative returns")

print("\n3. Win/Loss Ratio Analysis:")
print("   - 15m_basic: Best ratio (0.86) despite lower win rate")
print("   - 5m_basic: Lower ratio (0.63) but higher frequency compensates")
print("   - Need 56%+ win rate to be profitable with these ratios")

print("\n4. Statistical Significance:")
for name, data in results.items():
    trades = data['trades']
    wins = int(trades * data['win_rate'])
    # Simple binomial confidence interval
    z = 1.96  # 95% confidence
    p = data['win_rate']
    margin = z * ((p * (1-p)) / trades) ** 0.5
    print(f"   {name}: {p*100:.1f}% ± {margin*100:.1f}% (95% CI)")

print("\n=== CONCLUSION ===")
print("\nAll profitable strategies have 60%+ win rates:")
print("- 5m_basic: 65.4% (highest)")
print("- 5m_tuned: 63.5%") 
print("- 15m_basic: 61.9%")
print("\nThe 56.2% win rate of 15m_optimized is too low given the win/loss ratios.")