#!/usr/bin/env python3
"""Quick analysis of two-layer ensemble performance."""

import sys
sys.path.append('/Users/daws/ADMF-PC')

from pathlib import Path
from src.analytics.sparse_trace_analysis.strategy_analysis import StrategyAnalyzer
from src.analytics.sparse_trace_analysis.performance_calculation import ZERO_COST

# Workspace path
workspace_path = "./workspaces/two_layer_regime_ensemble_v1_c837887f"

# Initialize analyzer
strategy_analyzer = StrategyAnalyzer(workspace_path)

# Strategy files to analyze (they should all be identical)
strategy_files = [
    Path("SPY_baseline_plus_regime_boosters_{'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}}.parquet")
]

# Classifier to use
classifier_name = "SPY_market_regime_detector"

# Analyze strategies
results = strategy_analyzer.analyze_multiple_strategies(
    strategy_files=strategy_files,
    classifier_name=classifier_name,
    cost_config=ZERO_COST
)

if results:
    result = list(results.values())[0]
    
    print("=== TWO-LAYER ENSEMBLE ANALYSIS ===")
    print(f"Total trades: {result['total_trades']}")
    print(f"Cumulative return: {result['cumulative_return']:.2%}")
    print(f"Win rate: {result['win_rate']:.1%}")
    print(f"Avg trade return: {result['avg_trade_return']:.3%}")
    print(f"Trade frequency: 1 trade per {result['avg_bars_between_trades']:.0f} bars")
    
    print(f"\nRegime breakdown:")
    for regime, stats in result['regime_stats'].items():
        if stats['num_trades'] > 0:
            print(f"  {regime}: {stats['num_trades']} trades, {stats['cumulative_return']:.2%} return")