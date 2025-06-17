#!/usr/bin/env python3
"""
Analyze two-layer ensemble performance with realistic execution costs.
Tests multiple cost scenarios to understand strategy robustness.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.analytics.sparse_trace_analysis.strategy_analysis import StrategyAnalyzer
from src.analytics.sparse_trace_analysis.performance_calculation import ExecutionCostConfig

def analyze_with_costs():
    workspace_path = Path("workspaces/two_layer_regime_ensemble_v1_4f71d9e1")
    
    if not workspace_path.exists():
        print(f"❌ Workspace not found: {workspace_path}")
        return
    
    print("="*90)
    print("TWO-LAYER ENSEMBLE ANALYSIS WITH EXECUTION COSTS")
    print("="*90)
    
    # Initialize analyzer
    strategy_analyzer = StrategyAnalyzer(workspace_path)
    classifier_name = "SPY_market_regime_detector"
    
    # Find ensemble strategy files
    strategy_files = strategy_analyzer.find_strategy_files(
        strategy_pattern="baseline_plus_regime_boosters"
    )
    
    if not strategy_files:
        print("❌ No ensemble strategy files found")
        return
    
    print(f"Found {len(strategy_files)} ensemble strategies")
    print(f"Using classifier: {classifier_name}\n")
    
    # Define cost scenarios
    cost_scenarios = [
        ("Zero Cost (Baseline)", ExecutionCostConfig()),
        
        ("HFT Costs (5 bps RT)", ExecutionCostConfig(
            cost_multiplier=0.9995  # 0.05% per trade = 5 bps
        )),
        
        ("Institutional (10 bps RT)", ExecutionCostConfig(
            cost_multiplier=0.999   # 0.1% per trade = 10 bps
        )),
        
        ("Retail Conservative (20 bps RT)", ExecutionCostConfig(
            cost_multiplier=0.998   # 0.2% per trade = 20 bps
        )),
        
        ("Retail Realistic (30 bps RT)", ExecutionCostConfig(
            cost_multiplier=0.997   # 0.3% per trade = 30 bps
        )),
    ]
    
    # Analyze first strategy with different cost scenarios
    test_strategy = strategy_files[0]  # They're all identical anyway
    
    print(f"📊 COST IMPACT ANALYSIS")
    print(f"Strategy: {test_strategy.stem}")
    print("-" * 90)
    
    results_summary = []
    
    for scenario_name, cost_config in cost_scenarios:
        print(f"\n🔍 Analyzing: {scenario_name}")
        
        # Run analysis
        results = strategy_analyzer.analyze_multiple_strategies(
            strategy_files=[test_strategy],
            classifier_name=classifier_name,
            cost_config=cost_config
        )
        
        if results['strategies']:
            strategy_name = list(results['strategies'].keys())[0]
            strategy_data = results['strategies'][strategy_name]
            overall_perf = strategy_data['overall_performance']
            
            # Store results
            results_summary.append({
                'scenario': scenario_name,
                'trades': overall_perf['num_trades'],
                'gross_return': overall_perf.get('gross_percentage_return', overall_perf['percentage_return']),
                'net_return': overall_perf['percentage_return'],
                'win_rate': overall_perf['win_rate'],
                'max_drawdown': overall_perf.get('max_drawdown_pct', 0)
            })
            
            print(f"  - Total trades: {overall_perf['num_trades']}")
            print(f"  - Gross return: {overall_perf.get('gross_percentage_return', overall_perf['percentage_return']):.2%}")
            print(f"  - Net return: {overall_perf['percentage_return']:.2%}")
            print(f"  - Win rate: {overall_perf['win_rate']:.2%}")
            print(f"  - Max drawdown: {overall_perf.get('max_drawdown_pct', 0):.2%}")
    
    # Summary table
    print(f"\n📈 COST IMPACT SUMMARY")
    print("-" * 90)
    print(f"{'Scenario':<30} {'Gross Return':<12} {'Net Return':<12} {'Cost Impact':<12} {'Win Rate':<10}")
    print("-" * 90)
    
    baseline_return = results_summary[0]['net_return']
    
    for result in results_summary:
        cost_impact = result['gross_return'] - result['net_return']
        print(f"{result['scenario']:<30} {result['gross_return']:>11.2%} {result['net_return']:>11.2%} "
              f"{cost_impact:>11.2%} {result['win_rate']:>9.2%}")
    
    # Annualized returns (assuming ~52 trading days)
    print(f"\n📊 ANNUALIZED RETURNS (52 trading days)")
    print("-" * 90)
    print(f"{'Scenario':<30} {'Cumulative':<12} {'Annualized':<12}")
    print("-" * 90)
    
    for result in results_summary:
        cumulative = result['net_return']
        # Annualize: (1 + return)^(252/52) - 1
        annualized = (1 + cumulative) ** (252/52.4) - 1
        print(f"{result['scenario']:<30} {cumulative:>11.2%} {annualized:>11.2%}")
    
    # Break-even analysis
    print(f"\n💡 BREAK-EVEN ANALYSIS")
    print("-" * 90)
    
    # Find the cost level where returns go negative
    for result in results_summary:
        if result['net_return'] < 0:
            print(f"⚠️  Strategy becomes unprofitable at: {result['scenario']}")
            break
    else:
        print(f"✅ Strategy remains profitable across all tested cost scenarios")
    
    # Trade frequency analysis
    trades_per_day = results_summary[0]['trades'] / 52.4
    print(f"\n📊 TRADING FREQUENCY")
    print(f"  - Total trades: {results_summary[0]['trades']}")
    print(f"  - Trades per day: {trades_per_day:.1f}")
    print(f"  - Round trips per day: {trades_per_day/2:.1f}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 90)
    
    if results_summary[2]['net_return'] > 0:  # Institutional costs
        print("✅ Strategy is viable for institutional trading (10 bps costs)")
    
    if results_summary[3]['net_return'] > 0:  # Conservative retail
        print("✅ Strategy is viable for conservative retail trading (20 bps costs)")
    else:
        print("⚠️  Strategy may not be suitable for retail trading due to high frequency")
    
    optimal_annualized = (1 + results_summary[1]['net_return']) ** (252/52.4) - 1
    if optimal_annualized > 0.15:  # 15% annualized
        print(f"✅ Strong performance potential: {optimal_annualized:.1%} annualized at HFT costs")

if __name__ == "__main__":
    analyze_with_costs()