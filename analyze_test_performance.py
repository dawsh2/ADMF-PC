#!/usr/bin/env python3
"""Analyze two-layer ensemble performance on test dataset."""

import sys
sys.path.append('/Users/daws/ADMF-PC')

from pathlib import Path
import pandas as pd
from src.analytics.sparse_trace_analysis.strategy_analysis import StrategyAnalyzer, analyze_strategy_performance_by_regime
from src.analytics.sparse_trace_analysis.performance_calculation import ZERO_COST, ExecutionCostConfig

# Test workspace path
workspace_path = Path("./workspaces/two_layer_regime_ensemble_v1_3594c2a1")

# Initialize analyzer
strategy_analyzer = StrategyAnalyzer(workspace_path)

# Get the actual strategy file name from the traces directory
strategy_dir = workspace_path / "traces/SPY_1m/signals/ma_crossover"
strategy_files = list(strategy_dir.glob("SPY_baseline_plus_regime_boosters_*.parquet"))[:1]

# Classifier to use
classifier_name = "SPY_market_regime_detector"

print("=== TWO-LAYER ENSEMBLE TEST DATASET ANALYSIS ===")
print(f"Workspace: {workspace_path}")
print(f"Total bars: 102,235")
print(f"Classifier changes: 2,419")
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
        print(f"Annualized return: {result['annualized_return']:.2%}")
        print(f"Win rate: {result['win_rate']:.1%}")
        print(f"Avg trade return: {result['avg_trade_return']:.3%}")
        print(f"Trade frequency: 1 trade per {result['avg_bars_between_trades']:.0f} bars")
        
        print(f"\nRegime breakdown:")
        for regime, stats in result['regime_stats'].items():
            if stats['num_trades'] > 0:
                print(f"  {regime}: {stats['num_trades']} trades, {stats['cumulative_return']:.2%} return")

# Now with execution costs
print("\n\n2. WITH EXECUTION COSTS (10 bps):")
institutional_cost = ExecutionCostConfig(
    cost_model='multiplicative',
    cost_bps=10.0
)

results_with_cost = strategy_analyzer.analyze_multiple_strategies(
    strategy_files=strategy_files,
    classifier_name=classifier_name,
    cost_config=institutional_cost
)

if results_with_cost and 'strategies' in results_with_cost:
    for strategy_name, result in results_with_cost['strategies'].items():
        print(f"\nTotal trades: {result['total_trades']}")
        print(f"Cumulative return: {result['cumulative_return']:.2%}")
        print(f"Annualized return: {result['annualized_return']:.2%}")
        print(f"Execution costs: {result['total_execution_cost']:.2%}")
        print(f"Win rate: {result['win_rate']:.1%}")
        
# Verify signal/classifier ratio
print(f"\n\n3. SIGNAL ANALYSIS:")
print(f"Signal changes per classifier change: {3390 / 2419:.2f}")
print(f"Average regime duration: {102235 / 2419:.1f} bars ({102235 / 2419 / 60:.1f} hours)")
print(f"Average position duration: {102235 / (3390/2):.1f} bars ({102235 / (3390/2) / 60:.1f} hours)")