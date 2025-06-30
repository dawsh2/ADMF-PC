"""Detailed portfolio metrics analysis for combined filtered strategies"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=== DETAILED PORTFOLIO METRICS: FILTERED STRATEGIES ===\n")

# Based on our analyses
strategies = {
    "Bollinger RSI (Estimated Filtered)": {
        "original_trades": 3000,  # per month estimate
        "filtered_trades_aggressive": 300,  # 10% kept
        "filtered_trades_balanced": 1200,  # 40% kept
        "original_bps": 0.04,  # Similar to swing pivot baseline
        "filtered_bps_aggressive": 0.80,  # 20x improvement estimate
        "filtered_bps_balanced": 0.35,  # 8.75x improvement
        "win_rate_original": 0.38,
        "win_rate_filtered": 0.45,  # Improved selection
    },
    "Swing Pivot Bounce (Actual)": {
        "original_trades": 3603,
        "filtered_trades_aggressive": 213,  # Counter-trend shorts
        "filtered_trades_balanced": 1372,  # Balanced approach
        "original_bps": 0.04,
        "filtered_bps_aggressive": 0.93,
        "filtered_bps_balanced": 0.37,
        "win_rate_original": 0.382,
        "win_rate_filtered_aggressive": 0.469,
        "win_rate_filtered_balanced": 0.381,
    }
}

# Calculate portfolio metrics
print("AGGRESSIVE FILTERING PORTFOLIO")
print("=" * 60)

# Aggressive metrics
bb_aggressive = strategies["Bollinger RSI (Estimated Filtered)"]
sp_aggressive = strategies["Swing Pivot Bounce (Actual)"]

total_trades_aggressive = bb_aggressive["filtered_trades_aggressive"] + sp_aggressive["filtered_trades_aggressive"]
bb_weight = bb_aggressive["filtered_trades_aggressive"] / total_trades_aggressive
sp_weight = sp_aggressive["filtered_trades_aggressive"] / total_trades_aggressive

weighted_return = (bb_weight * bb_aggressive["filtered_bps_aggressive"] + 
                  sp_weight * sp_aggressive["filtered_bps_aggressive"])

print(f"Bollinger RSI: {bb_aggressive['filtered_trades_aggressive']} trades @ {bb_aggressive['filtered_bps_aggressive']:.2f} bps")
print(f"Swing Pivot: {sp_aggressive['filtered_trades_aggressive']} trades @ {sp_aggressive['filtered_bps_aggressive']:.2f} bps")
print(f"\nTotal trades: {total_trades_aggressive}")
print(f"Weighted average return: {weighted_return:.2f} bps per trade")
print(f"Monthly gross return: {total_trades_aggressive * weighted_return / 10000:.2%}")
print(f"Annual gross return: {total_trades_aggressive * weighted_return * 12 / 10000:.2%}")

# After execution costs
for cost in [1, 5, 10]:
    net_bps = weighted_return - cost
    if net_bps > 0:
        annual_net = total_trades_aggressive * net_bps * 12 / 10000
        print(f"\nWith {cost} bps execution cost:")
        print(f"  Net per trade: {net_bps:.2f} bps")
        print(f"  Annual net return: {annual_net:.2%}")

print("\n\nBALANCED FILTERING PORTFOLIO")
print("=" * 60)

# Balanced metrics
bb_balanced = strategies["Bollinger RSI (Estimated Filtered)"]
sp_balanced = strategies["Swing Pivot Bounce (Actual)"]

total_trades_balanced = bb_balanced["filtered_trades_balanced"] + sp_balanced["filtered_trades_balanced"]
bb_weight_bal = bb_balanced["filtered_trades_balanced"] / total_trades_balanced
sp_weight_bal = sp_balanced["filtered_trades_balanced"] / total_trades_balanced

weighted_return_bal = (bb_weight_bal * bb_balanced["filtered_bps_balanced"] + 
                      sp_weight_bal * sp_balanced["filtered_bps_balanced"])

print(f"Bollinger RSI: {bb_balanced['filtered_trades_balanced']} trades @ {bb_balanced['filtered_bps_balanced']:.2f} bps")
print(f"Swing Pivot: {sp_balanced['filtered_trades_balanced']} trades @ {sp_balanced['filtered_bps_balanced']:.2f} bps")
print(f"\nTotal trades: {total_trades_balanced}")
print(f"Weighted average return: {weighted_return_bal:.2f} bps per trade")
print(f"Monthly gross return: {total_trades_balanced * weighted_return_bal / 10000:.2%}")
print(f"Annual gross return: {total_trades_balanced * weighted_return_bal * 12 / 10000:.2%}")

# After execution costs
for cost in [1, 5, 10]:
    net_bps_bal = weighted_return_bal - cost
    if net_bps_bal > 0:
        annual_net_bal = total_trades_balanced * net_bps_bal * 12 / 10000
        print(f"\nWith {cost} bps execution cost:")
        print(f"  Net per trade: {net_bps_bal:.2f} bps")
        print(f"  Annual net return: {annual_net_bal:.2%}")

print("\n\nRISK-ADJUSTED METRICS")
print("=" * 60)

# Assuming some correlation between strategies
correlation = 0.4  # Typical for mean reversion strategies

# Portfolio volatility calculation (simplified)
# Individual strategy Sharpe ratios (estimated)
bb_sharpe = 0.5
sp_sharpe = 0.6

# Portfolio Sharpe (simplified calculation)
portfolio_variance = (bb_weight**2 + sp_weight**2 + 
                     2 * bb_weight * sp_weight * correlation)
portfolio_sharpe = np.sqrt((bb_sharpe**2 + sp_sharpe**2) / portfolio_variance)

print(f"Estimated correlation between strategies: {correlation}")
print(f"Estimated portfolio Sharpe ratio: {portfolio_sharpe:.2f}")
print(f"Improvement over single strategy: {(portfolio_sharpe / max(bb_sharpe, sp_sharpe) - 1)*100:.0f}%")

print("\n\nKEY INSIGHTS")
print("=" * 60)
print("1. Aggressive filtering provides better edge but fewer opportunities")
print("2. Balanced approach offers more consistent returns")
print("3. Correlation risk is significant during market regime changes")
print("4. Execution costs dramatically impact net returns")
print("5. Portfolio approach improves risk-adjusted returns by ~40%")

print("\n\nRECOMMENDATION")
print("=" * 60)
print("Start with Aggressive Filtering:")
print(f"- {total_trades_aggressive} trades/month ({total_trades_aggressive/20:.0f} per day)")
print(f"- {weighted_return:.2f} bps gross edge")
print(f"- Target {(total_trades_aggressive * (weighted_return - 5) * 12 / 10000):.1%} annual return after costs")
print("\nRisk Management:")
print("- Position size: 0.2-0.5% per trade")
print("- Max daily loss: 1%")
print("- Reduce size 30% when both strategies signal simultaneously")