#!/usr/bin/env python3
"""Properly analyze two-layer ensemble performance on test dataset."""

import sys
sys.path.append('/Users/daws/ADMF-PC')

from pathlib import Path
from src.analytics.sparse_trace_analysis.strategy_analysis import StrategyAnalyzer
from src.analytics.sparse_trace_analysis.performance_calculation import ZERO_COST, ExecutionCostConfig

# Test workspace path
workspace_path = Path("./workspaces/two_layer_regime_ensemble_v1_3594c2a1")

# Initialize analyzer
strategy_analyzer = StrategyAnalyzer(str(workspace_path))

# Use the actual strategy name without .parquet extension
strategy_files = [
    Path("SPY_baseline_plus_regime_boosters_{'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}}")
]

# Classifier to use
classifier_name = "SPY_market_regime_detector"

print("=== TWO-LAYER ENSEMBLE TEST DATASET ANALYSIS ===")
print(f"Workspace: {workspace_path}")
print(f"Test period: Bars 81,812 - 102,235 (20,423 bars)")
print(f"Classifier changes: 2,419 over full period")
print(f"Signal changes: 3,390 per strategy\n")

# First analyze with zero cost
print("1. ZERO COST ANALYSIS:")
results = strategy_analyzer.analyze_multiple_strategies(
    strategy_files=strategy_files,
    classifier_name=classifier_name,
    cost_config=ZERO_COST
)

if results and 'strategies' in results:
    for strategy_name, result in results['strategies'].items():
        print(f"\nTotal trades: {result['total_trades']}")
        print(f"Cumulative return: {result['cumulative_return']:.2%}")
        print(f"Win rate: {result['win_rate']:.1%}")
        print(f"Avg trade return: {result['avg_trade_return']:.3%}")
        print(f"Sharpe ratio: {result.get('sharpe_ratio', 'N/A')}")
        print(f"Max drawdown: {result.get('max_drawdown', 'N/A')}")
        
        # Calculate annualized return
        # 20,423 bars at 1-minute = ~340 hours = ~42.5 trading days
        trading_days = 20423 / 60 / 8  # Assuming 8-hour trading days
        years = trading_days / 252
        if years > 0:
            annualized = (1 + result['cumulative_return']) ** (1/years) - 1
            print(f"Annualized return: {annualized:.2%}")
        
        print(f"\nRegime breakdown:")
        for regime, stats in result['regime_stats'].items():
            if stats['num_trades'] > 0:
                print(f"  {regime}: {stats['num_trades']} trades, {stats['cumulative_return']:.2%} return")

# Now with execution costs
print("\n\n2. WITH EXECUTION COSTS (10 bps):")
institutional_cost = ExecutionCostConfig(
    model='multiplicative',
    bps=10.0
)

results_with_cost = strategy_analyzer.analyze_multiple_strategies(
    strategy_files=strategy_files,
    classifier_name=classifier_name,
    cost_config=institutional_cost
)

if results_with_cost and 'strategies' in results_with_cost:
    for strategy_name, result in results_with_cost['strategies'].items():
        print(f"\nCumulative return after costs: {result['cumulative_return']:.2%}")
        print(f"Execution costs: {result.get('total_execution_cost', 0):.2%}")
        
        # Calculate annualized return after costs
        if years > 0:
            annualized_after_costs = (1 + result['cumulative_return']) ** (1/years) - 1
            print(f"Annualized return after costs: {annualized_after_costs:.2%}")

# Analyze signal patterns
print(f"\n\n3. SIGNAL & REGIME ANALYSIS:")
print(f"Average regime duration: {102235 / 2419:.1f} bars ({102235 / 2419 / 60:.1f} hours)")
print(f"Average position duration: {20423 / (3390/2):.1f} bars ({20423 / (3390/2) / 60:.1f} hours)")
print(f"Signals per regime change: {3390 / 2419:.2f}")