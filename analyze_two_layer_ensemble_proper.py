#!/usr/bin/env python3
"""
Proper analysis of two-layer ensemble using sparse trace analysis tools.
This correctly handles the sparse storage format where signals represent changes only.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.analytics.sparse_trace_analysis.strategy_analysis import StrategyAnalyzer
from src.analytics.sparse_trace_analysis.classifier_analysis import ClassifierAnalyzer
from src.analytics.sparse_trace_analysis.performance_calculation import ZERO_COST, INSTITUTIONAL
from src.analytics.sparse_trace_analysis.regime_attribution import RegimeAttributor

def main():
    workspace_path = Path("workspaces/two_layer_regime_ensemble_v1_4f71d9e1")
    
    if not workspace_path.exists():
        print(f"❌ Workspace not found: {workspace_path}")
        return
    
    print("="*90)
    print("PROPER TWO-LAYER ENSEMBLE ANALYSIS")
    print("Using sparse trace analysis tools that understand the signal format")
    print("="*90)
    
    # 1. Analyze classifier performance first
    print("\n🔍 CLASSIFIER ANALYSIS")
    print("-" * 50)
    
    classifier_analyzer = ClassifierAnalyzer(workspace_path)
    classifier_analysis = classifier_analyzer.analyze_all_classifiers()
    
    if not classifier_analysis:
        print("❌ No classifier data found or analysis failed")
        return
    
    # Print classifier analysis
    for classifier_name, analysis in classifier_analysis.items():
        print(f"\nClassifier: {classifier_name}")
        print(f"  - Analysis keys: {list(analysis.keys())}")
        
        # Handle different possible structures
        if 'num_state_changes' in analysis:
            print(f"  - State changes: {analysis['num_state_changes']}")
        elif 'state_changes' in analysis:
            print(f"  - State changes: {len(analysis['state_changes'])}")
        
        if 'state_durations' in analysis:
            print(f"  - Unique states: {len(analysis['state_durations'])}")
            print(f"  - State distribution:")
            
            total_duration = sum(analysis['state_durations'].values())
            for state, duration in analysis['state_durations'].items():
                percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
                print(f"    {state}: {percentage:.1f}% ({duration} bars)")
        
        # Print balance metrics
        if 'balance_score' in analysis:
            print(f"  - Balance score: {analysis['balance_score']:.1f}")
        if 'normalized_entropy' in analysis:
            print(f"  - Normalized entropy: {analysis['normalized_entropy']:.3f}")
        
        # Print raw analysis for debugging
        print(f"  - Full analysis: {analysis}")
    
    # Select the classifier we know exists (from the analysis output above)
    classifier_name = "SPY_market_regime_detector"  # This is the actual classifier file name
    
    # 2. Analyze strategy performance using proper sparse tools
    print(f"\n📊 STRATEGY PERFORMANCE ANALYSIS")
    print(f"Using classifier: {classifier_name}")
    print("-" * 50)
    
    strategy_analyzer = StrategyAnalyzer(workspace_path)
    
    # Find all ensemble strategy files
    strategy_files = strategy_analyzer.find_strategy_files(
        strategy_pattern="baseline_plus_regime_boosters"
    )
    
    print(f"Found {len(strategy_files)} ensemble strategy files:")
    for f in strategy_files:
        print(f"  - {f.name}")
    
    if not strategy_files:
        print("❌ No ensemble strategy files found")
        return
    
    # Analyze strategies with zero cost first
    print(f"\n🧮 ANALYZING PERFORMANCE (Zero Cost)")
    print("-" * 50)
    
    results = strategy_analyzer.analyze_multiple_strategies(
        strategy_files=strategy_files,
        classifier_name=classifier_name,
        cost_config=ZERO_COST
    )
    
    # Print overall summary
    successful_analyses = len(results['strategies'])
    print(f"\n✅ Successfully analyzed {successful_analyses} strategies")
    
    if successful_analyses == 0:
        print("❌ No strategies could be analyzed")
        return
    
    # 3. Detailed performance analysis
    print(f"\n📈 DETAILED PERFORMANCE BREAKDOWN")
    print("-" * 50)
    
    for strategy_name, strategy_data in results['strategies'].items():
        print(f"\nStrategy: {strategy_name}")
        
        overall_perf = strategy_data['overall_performance']
        print(f"  Overall Performance:")
        print(f"    - Total trades: {overall_perf['num_trades']}")
        print(f"    - Total return: {overall_perf['percentage_return']:.2%}")
        print(f"    - Win rate: {overall_perf['win_rate']:.2%}")
        print(f"    - Avg trade return: {overall_perf['avg_trade_log_return']:.4f}")
        
        # Regime breakdown
        regime_perf = strategy_data.get('regime_performance', {})
        if regime_perf:
            print(f"  Performance by Regime:")
            for regime, perf in regime_perf.items():
                print(f"    {regime}:")
                print(f"      - Trades: {perf['trade_count']}")
                print(f"      - Return: {perf['net_percentage_return']:.2%}")
                print(f"      - Win rate: {perf['win_rate']:.2%}")
        else:
            print(f"  ⚠️  No regime attribution available")
        
        # Attribution accuracy
        attribution = strategy_data.get('regime_attribution', {})
        if 'attribution_accuracy' in attribution:
            print(f"  Regime Attribution: {attribution['attribution_accuracy']:.2%} accuracy")
    
    # 4. Compare strategies by regime
    print(f"\n🏆 TOP PERFORMERS BY REGIME")
    print("-" * 50)
    
    # Get unique regimes from results
    all_regimes = set()
    for strategy_data in results['strategies'].values():
        all_regimes.update(strategy_data.get('regime_performance', {}).keys())
    
    for regime in sorted(all_regimes):
        strategy_analyzer.print_strategy_comparison(
            results, regime, 'net_percentage_return', top_n=3
        )
    
    # 5. Check for signal diversity
    print(f"\n🔄 SIGNAL DIVERSITY ANALYSIS")
    print("-" * 50)
    
    # Load one strategy to check signal patterns
    if strategy_files:
        from src.analytics.sparse_trace_analysis.strategy_analysis import load_strategy_signals
        
        print("Checking signal patterns (sparse format understanding):")
        
        for i, strategy_file in enumerate(strategy_files[:3]):  # Check first 3
            signals_df = load_strategy_signals(strategy_file)
            
            if signals_df is not None:
                print(f"\n  Strategy {i+1}: {strategy_file.stem}")
                print(f"    - Signal changes: {len(signals_df)}")
                print(f"    - Bar range: {signals_df['bar_idx'].min()} - {signals_df['bar_idx'].max()}")
                print(f"    - Signal values: {sorted(signals_df['signal_value'].unique())}")
                print(f"    - Price range: ${signals_df['price'].min():.2f} - ${signals_df['price'].max():.2f}")
                
                # Show signal transition pattern
                transitions = []
                for j in range(min(5, len(signals_df) - 1)):
                    current = signals_df.iloc[j]
                    next_sig = signals_df.iloc[j + 1]
                    transitions.append(f"{current['signal_value']}@${current['price']:.0f} → {next_sig['signal_value']}@${next_sig['price']:.0f}")
                
                if transitions:
                    print(f"    - First 5 transitions: {'; '.join(transitions)}")
    
    # 6. Generate summary report
    print(f"\n📋 GENERATING SUMMARY REPORT")
    print("-" * 50)
    
    report = strategy_analyzer.generate_regime_summary_report(results)
    
    # Save results
    output_file = workspace_path / "proper_analysis_results.json"
    strategy_analyzer.save_analysis_results(results, output_file)
    
    # Save report
    report_file = workspace_path / "proper_analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Analysis complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📄 Report saved to: {report_file}")
    
    # 7. Key insights
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 50)
    
    if successful_analyses > 1:
        # Check if strategies are producing different signals
        total_trades = [data['overall_performance']['num_trades'] for data in results['strategies'].values()]
        returns = [data['overall_performance']['percentage_return'] for data in results['strategies'].values()]
        
        if len(set(total_trades)) == 1:
            print("⚠️  All strategies have identical trade counts - potential signal duplication")
        else:
            print("✅ Strategies show different trade patterns")
        
        if len(set([round(r, 4) for r in returns])) == 1:
            print("⚠️  All strategies have identical returns - likely identical signals")
        else:
            print("✅ Strategies show different performance")
        
        print(f"📊 Trade count range: {min(total_trades)} - {max(total_trades)}")
        print(f"📊 Return range: {min(returns):.2%} - {max(returns):.2%}")
    
    print(f"\n🎯 SPARSE TRACE FORMAT CONFIRMATION")
    print("-" * 50)
    print("✅ Analysis used proper sparse trace understanding:")
    print("  - Signals represent position changes, not every bar")
    print("  - First signal opens position")
    print("  - Subsequent signals close/flip positions")
    print("  - Only meaningful changes are stored")
    print("  - Regime attribution based on position opening")

if __name__ == "__main__":
    main()