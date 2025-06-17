#!/usr/bin/env python3
"""Analyze two-layer ensemble with execution costs using multiplicative model."""

import sys
sys.path.append('/Users/daws/ADMF-PC')

from pathlib import Path
from src.analytics.sparse_trace_analysis.strategy_analysis import analyze_strategy_performance_by_regime
from src.analytics.sparse_trace_analysis.performance_calculation import ExecutionCostConfig, ZERO_COST

# Test workspace
workspace_path = Path("./workspaces/two_layer_regime_ensemble_v1_3594c2a1")

# Build the correct strategy file path
strategy_filename = "SPY_baseline_plus_regime_boosters_{'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}}.parquet"
strategy_file = workspace_path / "traces/SPY_1m/signals/ma_crossover" / strategy_filename
classifier_name = "SPY_market_regime_detector"

print("=== TWO-LAYER ENSEMBLE PERFORMANCE WITH COSTS ===\n")
print("Test period: Bars 81,812 - 102,235 (20,423 bars)")
print("Using multiplicative cost model (works naturally with log returns)\n")

# Define cost scenarios
cost_scenarios = [
    ("Zero Cost", ZERO_COST),
    ("HFT (5 bps)", ExecutionCostConfig(cost_multiplier=0.9995)),  # 5 bps = 0.05% = multiply by 0.9995
    ("Institutional (10 bps)", ExecutionCostConfig(cost_multiplier=0.999)),  # 10 bps = 0.1% = multiply by 0.999
    ("Retail (20 bps)", ExecutionCostConfig(cost_multiplier=0.998)),  # 20 bps = 0.2% = multiply by 0.998
    ("High Cost (50 bps)", ExecutionCostConfig(cost_multiplier=0.995))  # 50 bps = 0.5% = multiply by 0.995
]

results = {}

for scenario_name, cost_config in cost_scenarios:
    print(f"\n{'='*60}")
    print(f"{scenario_name}")
    print('='*60)
    
    result = analyze_strategy_performance_by_regime(
        strategy_file,
        classifier_name, 
        workspace_path,
        cost_config
    )
    
    if result:
        results[scenario_name] = result
        
        # Debug: print available keys
        if scenario_name == "Zero Cost":
            print(f"\nDEBUG - Result structure:")
            print(f"Top-level keys: {list(result.keys())}")
            if 'overall_performance' in result:
                print(f"overall_performance keys: {list(result['overall_performance'].keys())}")
        
        # Core metrics - use the correct keys from overall_performance
        overall_perf = result.get('overall_performance', {})
        print(f"Total trades: {overall_perf.get('num_trades', 0)}")
        print(f"Cumulative return: {overall_perf.get('percentage_return', 0):.2%}")
        print(f"Win rate: {overall_perf.get('win_rate', 0):.1%}")
        print(f"Avg trade return: {overall_perf.get('avg_trade_log_return', 0):.3%}")
        print(f"Total log return: {overall_perf.get('total_log_return', 0):.4f}")
        
        # Check for Sharpe ratio in results
        if 'sharpe_ratio' in overall_perf:
            print(f"Sharpe ratio: {overall_perf['sharpe_ratio']:.2f}")
        else:
            # Calculate approximate Sharpe if we have trade data
            trades = overall_perf.get('trades', [])
            if trades:
                import numpy as np
                trade_returns = [t.get('log_return', 0) for t in trades if 'log_return' in t]
                if trade_returns:
                    # Annualize based on trade frequency
                    trades_per_day = overall_perf.get('num_trades', 0) / 42.5
                    annualization_factor = np.sqrt(252 * trades_per_day / overall_perf.get('num_trades', 1))
                    
                    avg_return = np.mean(trade_returns)
                    std_return = np.std(trade_returns)
                    if std_return > 0:
                        sharpe = (avg_return / std_return) * annualization_factor
                        print(f"Sharpe ratio (estimated): {sharpe:.2f}")
                    else:
                        print(f"Sharpe ratio: N/A (no volatility)")
        
        # Annualize (20,423 bars = ~42.5 trading days = ~0.17 years)
        years = 20423 / 60 / 8 / 252
        cum_return = overall_perf.get('percentage_return', 0)
        annualized = (1 + cum_return) ** (1/years) - 1 if cum_return > -1 else 0
        print(f"Annualized return: {annualized:.2%}")
        
        # Regime breakdown
        regime_perf = result.get('regime_performance', {})
        if regime_perf:
            print(f"\nRegime performance:")
            for regime, stats in regime_perf.items():
                if stats.get('num_trades', 0) > 0:
                    print(f"  {regime}: {stats['num_trades']} trades, {stats.get('percentage_return', 0):.2%} return")
    else:
        print("Analysis failed!")

# Summary comparison
if len(results) > 1:
    print(f"\n\n{'='*60}")
    print("COST IMPACT SUMMARY")
    print('='*60)
    
    zero_cost_return = results.get("Zero Cost", {}).get('overall_performance', {}).get('percentage_return', 0)
    zero_cost_trades = results.get("Zero Cost", {}).get('overall_performance', {}).get('num_trades', 0)
    
    print(f"{'Scenario':<20} {'Return':>10} {'Cost Impact':>12} {'Annualized':>12}")
    print("-" * 56)
    
    for scenario_name, result in results.items():
        overall_perf = result.get('overall_performance', {})
        cum_return = overall_perf.get('percentage_return', 0)
        cost_impact = cum_return - zero_cost_return if scenario_name != "Zero Cost" else 0
        
        years = 20423 / 60 / 8 / 252
        annualized = (1 + cum_return) ** (1/years) - 1 if cum_return > -1 else 0
        
        print(f"{scenario_name:<20} {cum_return:>10.2%} {cost_impact:>12.2%} {annualized:>12.2%}")
    
    print(f"\nKey insights:")
    print(f"- Strategy generates {zero_cost_trades} trades over ~42 trading days")
    print(f"- That's ~{zero_cost_trades / 42.5:.1f} trades per day")
    
    # Check if profitable after costs
    inst_return = results.get("Institutional (10 bps)", {}).get('overall_performance', {}).get('percentage_return', 0)
    if inst_return > 0:
        print(f"- Strategy remains profitable after institutional costs ({inst_return:.2%})")
    else:
        print(f"- Strategy becomes unprofitable at institutional cost levels")
        
    # Break-even analysis
    if zero_cost_return > 0 and zero_cost_trades > 0:
        breakeven_multiplier = 1 / (1 + zero_cost_return) ** (1 / zero_cost_trades)
        breakeven_bps = (1 - breakeven_multiplier) * 10000
        print(f"- Break-even cost level: ~{breakeven_bps:.0f} bps per trade")