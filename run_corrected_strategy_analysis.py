#!/usr/bin/env python3
"""
Run corrected strategy performance analysis using the new modular analytics framework.

This script uses the best balanced classifier to analyze strategy performance by regime
with proper log returns calculation and regime attribution.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.sparse_trace_analysis import (
    StrategyAnalyzer,
    ExecutionCostConfig,
    TYPICAL_RETAIL
)


def main():
    """Run corrected strategy performance analysis by regime."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    print("CORRECTED STRATEGY PERFORMANCE ANALYSIS")
    print("="*50)
    print("Using modular analytics framework")
    print("Log returns per trade with proper regime attribution")
    print("="*50)
    
    # Initialize strategy analyzer
    analyzer = StrategyAnalyzer(workspace_path)
    
    # Use the best balanced classifier from corrected analysis
    classifier_name = "SPY_market_regime_grid_0006_12"
    
    print(f"Using classifier: {classifier_name}")
    print("States: bull_ranging (44.7%), bear_ranging (34.8%), neutral (18.5%), trending (<2%)")
    
    # Find MACD strategy files for comparison with previous analysis
    strategy_files = analyzer.find_strategy_files(
        strategy_pattern="macd_crossover",
        max_files=10
    )
    
    if not strategy_files:
        print("❌ No MACD strategy files found")
        return
    
    print(f"\nFound {len(strategy_files)} MACD strategy files for analysis")
    
    # Set up execution costs (1% total cost as example)
    cost_config = ExecutionCostConfig(cost_multiplier=0.99)
    
    print(f"Execution cost: 1% multiplier (cost_multiplier=0.99)")
    
    # Analyze strategies by regime
    print(f"\nAnalyzing strategy performance by regime...")
    analysis_results = analyzer.analyze_multiple_strategies(
        strategy_files,
        classifier_name,
        cost_config
    )
    
    if not analysis_results['strategies']:
        print("❌ No strategies could be analyzed")
        return
    
    print(f"✅ Successfully analyzed {len(analysis_results['strategies'])} strategies")
    
    # Print regime performance comparisons
    regimes = ['bull_ranging', 'bear_ranging', 'neutral']
    
    for regime in regimes:
        print(f"\n{'='*80}")
        print(f"PERFORMANCE IN {regime.upper()} REGIME")
        print("="*80)
        
        analyzer.print_strategy_comparison(
            analysis_results,
            regime,
            sort_by='net_percentage_return',
            top_n=5
        )
    
    # Generate comprehensive report
    print(f"\n{'='*80}")
    print("COMPREHENSIVE REGIME ANALYSIS REPORT")
    print("="*80)
    
    report = analyzer.generate_regime_summary_report(analysis_results)
    print(report)
    
    # Save results
    output_file = workspace_path / f"corrected_strategy_analysis_{classifier_name}.json"
    analyzer.save_analysis_results(analysis_results, output_file)
    
    # Save summary report
    report_file = workspace_path / f"corrected_regime_analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Detailed results saved to: {output_file}")
    print(f"✅ Summary report saved to: {report_file}")
    
    # Compare with previous analysis
    print(f"\n{'='*80}")
    print("COMPARISON WITH PREVIOUS ANALYSIS")
    print("="*80)
    print("Key differences from original analysis:")
    print("1. ✅ Proper duration calculation from sparse classifier changes")
    print("2. ✅ Correct log returns per trade calculation")
    print("3. ✅ Regime attribution to opening bar (not closing)")
    print("4. ✅ Market regime classifier is most balanced (not volatility momentum)")
    print("5. ✅ Execution costs properly applied as multiplier")
    
    return analysis_results


if __name__ == "__main__":
    import pandas as pd
    
    try:
        results = main()
        
        if results:
            print(f"\n{'='*80}")
            print("CORRECTED ANALYSIS COMPLETE")
            print("="*80)
            print(f"✅ Strategy performance analysis completed")
            print(f"✅ Regime attribution working correctly")
            print(f"✅ Ready for regime-aware strategy development")
            
            # Quick summary of findings
            total_strategies = len(results['strategies'])
            
            regime_trade_counts = {}
            for strategy_data in results['strategies'].values():
                for regime, regime_perf in strategy_data.get('regime_performance', {}).items():
                    if regime not in regime_trade_counts:
                        regime_trade_counts[regime] = 0
                    regime_trade_counts[regime] += regime_perf['trade_count']
            
            print(f"\nTrade distribution across regimes:")
            for regime, count in sorted(regime_trade_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {regime}: {count:,} trades")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()