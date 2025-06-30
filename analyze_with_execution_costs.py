"""Calculate strategy returns with realistic execution costs"""
import pandas as pd
import numpy as np

# Base parameters
execution_cost_bps = 1  # 1 basis point per side = 2 bps round trip
execution_cost_pct = execution_cost_bps / 10000  # Convert to percentage

print(f"=== Strategy Analysis with {execution_cost_bps} bps Execution Cost ===")
print(f"(Total round-trip cost: {execution_cost_bps * 2} bps or {execution_cost_pct * 2 * 100:.02f}%)\n")

# Strategy configurations
strategies = {
    'Original L/S': {
        'avg_return_gross': 0.015,
        'trades_per_year': 772
    },
    'Original + Stop': {
        'avg_return_gross': 0.034,
        'trades_per_year': 772
    },
    'VWAP Filter': {
        'avg_return_gross': 0.030,
        'trades_per_year': 536
    },
    'VWAP Filter + Stop': {
        'avg_return_gross': 0.039,
        'trades_per_year': 536
    },
    'All Filters + Stop': {
        'avg_return_gross': 0.033,
        'trades_per_year': 532
    }
}

# Calculate net returns
print(f"{'Strategy':<20} {'Gross/Trade':<12} {'Cost/Trade':<12} {'Net/Trade':<12} {'Trades/Yr':<10} {'Gross Annual':<12} {'Net Annual':<12}")
print("-" * 100)

results = []
for name, params in strategies.items():
    gross_per_trade = params['avg_return_gross'] / 100
    cost_per_trade = execution_cost_pct * 2  # Round trip
    net_per_trade = gross_per_trade - cost_per_trade
    trades_year = params['trades_per_year']
    
    # Annual returns
    gross_annual = (1 + gross_per_trade) ** trades_year - 1
    net_annual = (1 + net_per_trade) ** trades_year - 1 if net_per_trade > 0 else -1
    
    print(f"{name:<20} {gross_per_trade*100:>10.3f}% {cost_per_trade*100:>10.3f}% "
          f"{net_per_trade*100:>10.3f}% {trades_year:>9} {gross_annual*100:>10.1f}% "
          f"{net_annual*100:>10.1f}%")
    
    results.append({
        'strategy': name,
        'net_per_trade': net_per_trade,
        'net_annual': net_annual,
        'profitable': net_per_trade > 0
    })

# Analysis of results
print("\n=== Key Insights ===")

profitable_strategies = [r for r in results if r['profitable']]
if profitable_strategies:
    print(f"\nProfitable strategies after costs:")
    for r in sorted(profitable_strategies, key=lambda x: x['net_annual'], reverse=True):
        print(f"  {r['strategy']}: {r['net_annual']*100:.1f}% net annual")
else:
    print("\nNo strategies profitable after execution costs!")

# Break-even analysis
print("\n=== Break-Even Analysis ===")
print("Average return per trade needed to break even with 2 bps round-trip cost:")
print(f"  Minimum: {execution_cost_pct * 2 * 100:.3f}%")

for name, params in strategies.items():
    trades_year = params['trades_per_year']
    # What gross return per trade would give 10% annual after costs?
    target_annual = 0.10
    required_net_per_trade = (1 + target_annual) ** (1/trades_year) - 1
    required_gross_per_trade = required_net_per_trade + (execution_cost_pct * 2)
    
    print(f"\n{name} needs {required_gross_per_trade*100:.3f}% per trade for 10% annual")
    print(f"  Current: {params['avg_return_gross']:.3f}%")
    print(f"  Gap: {(required_gross_per_trade*100 - params['avg_return_gross']):.3f}%")

# Cost sensitivity analysis
print("\n=== Cost Sensitivity Analysis ===")
print("Net annual returns at different execution costs:\n")

cost_scenarios = [0.5, 1, 2, 5, 10]  # basis points per side
print(f"{'Strategy':<20}", end="")
for cost_bps in cost_scenarios:
    print(f"{cost_bps}bps".rjust(10), end="")
print()
print("-" * 70)

for name, params in strategies.items():
    print(f"{name:<20}", end="")
    for cost_bps in cost_scenarios:
        cost_pct = (cost_bps / 10000) * 2  # Round trip
        net_per_trade = params['avg_return_gross'] / 100 - cost_pct
        if net_per_trade > 0:
            net_annual = (1 + net_per_trade) ** params['trades_per_year'] - 1
            print(f"{net_annual*100:>8.1f}%", end="")
        else:
            print(f"{'LOSS':>9}", end="")
    print()

# Recommendations
print("\n=== Recommendations ===")
print(f"With {execution_cost_bps} bps execution costs:")

best_strategy = max(profitable_strategies, key=lambda x: x['net_annual']) if profitable_strategies else None
if best_strategy:
    print(f"\n1. Best strategy: {best_strategy['strategy']}")
    print(f"   Net annual return: {best_strategy['net_annual']*100:.1f}%")
    
    # Calculate required capital for meaningful returns
    target_annual_profit = 10000  # $10k annual profit
    required_capital = target_annual_profit / best_strategy['net_annual']
    print(f"\n2. To earn ${target_annual_profit:,} annually:")
    print(f"   Required capital: ${required_capital:,.0f}")
    
    print("\n3. Key success factors:")
    print("   - Stop loss is critical for profitability")
    print("   - Execution quality matters significantly")
    print("   - Consider limit orders to reduce costs")
else:
    print("\nNo strategies remain profitable after realistic execution costs!")
    print("Need to either:")
    print("  - Improve signal quality")
    print("  - Reduce execution costs")
    print("  - Increase holding periods")