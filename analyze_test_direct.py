#!/usr/bin/env python3
"""Direct analysis of test performance."""

import sys
sys.path.append('/Users/daws/ADMF-PC')

from pathlib import Path
from src.analytics.sparse_trace_analysis.strategy_analysis import analyze_strategy_performance_by_regime
from src.analytics.sparse_trace_analysis.performance_calculation import ZERO_COST, ExecutionCostConfig

# Paths
workspace_path = Path("./workspaces/two_layer_regime_ensemble_v1_3594c2a1")
strategy_file = "SPY_baseline_plus_regime_boosters_{'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}}"
classifier_name = "SPY_market_regime_detector"

print("=== TWO-LAYER ENSEMBLE TEST PERFORMANCE ===")
print(f"Test period: Last 20% of data (~20,423 bars)")
print(f"Classifier changes: 2,419")
print(f"Signal changes: 3,390\n")

# Zero cost analysis
print("1. WITHOUT EXECUTION COSTS:")
result = analyze_strategy_performance_by_regime(
    strategy_file, 
    classifier_name, 
    workspace_path,
    ZERO_COST
)

if result:
    print(f"Total trades: {result['total_trades']}")
    print(f"Cumulative return: {result['cumulative_return']:.2%}")
    print(f"Win rate: {result['win_rate']:.1%}")
    print(f"Avg trade return: {result['avg_trade_return']:.3%}")
    print(f"Best trade: {result['best_trade']:.2%}")
    print(f"Worst trade: {result['worst_trade']:.2%}")
    
    # Estimate annualized return
    # ~20,423 bars = ~340 hours = ~42.5 trading days = ~0.17 years
    years = 20423 / 60 / 8 / 252
    annualized = (1 + result['cumulative_return']) ** (1/years) - 1
    print(f"Annualized return (estimated): {annualized:.2%}")
    
    print(f"\nRegime performance:")
    for regime, stats in result['regime_stats'].items():
        if stats['num_trades'] > 0:
            print(f"  {regime}: {stats['num_trades']} trades, {stats['cumulative_return']:.2%} return, {stats['win_rate']:.1%} win rate")

# With costs
print("\n2. WITH 10 BPS EXECUTION COSTS:")
cost_config = ExecutionCostConfig(
    cost_model='multiplicative',
    cost_bps=10.0
)

result_with_costs = analyze_strategy_performance_by_regime(
    strategy_file,
    classifier_name,
    workspace_path, 
    cost_config
)

if result_with_costs:
    print(f"Cumulative return after costs: {result_with_costs['cumulative_return']:.2%}")
    print(f"Total execution cost impact: {result['cumulative_return'] - result_with_costs['cumulative_return']:.2%}")
    
    annualized_with_costs = (1 + result_with_costs['cumulative_return']) ** (1/years) - 1
    print(f"Annualized return after costs: {annualized_with_costs:.2%}")

# Additional metrics
print(f"\n3. TRADING PATTERNS:")
if result:
    print(f"Signals per regime: {3390 / 2419:.1f}")
    print(f"Avg bars per trade: {result['avg_bars_between_trades']:.0f} ({result['avg_bars_between_trades']/60:.1f} hours)")
    print(f"Trade frequency: {result['total_trades'] / (20423/60/8):.1f} trades per day")
    
    # Regime persistence
    print(f"\nRegime persistence:")
    print(f"Avg regime duration: {102235 / 2419:.0f} bars ({102235 / 2419 / 60:.1f} hours)")
    print(f"Test period regime changes: ~{int(2419 * 0.2)} (estimated from 20% of data)")