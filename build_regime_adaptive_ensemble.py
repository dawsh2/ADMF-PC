#!/usr/bin/env python3
"""
Build regime-adaptive ensemble by finding top performers per regime.

This script analyzes the full strategy universe to identify the best performing
strategies in each market regime, then constructs an adaptive ensemble.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.sparse_trace_analysis import (
    StrategyAnalyzer,
    ExecutionCostConfig,
    ClassifierAnalyzer
)


def analyze_full_strategy_universe(
    workspace_path: Path,
    classifier_name: str,
    max_strategies: int = None,
    cost_config: ExecutionCostConfig = None
):
    """
    Analyze all available strategies to find regime-specific top performers.
    
    Args:
        workspace_path: Path to workspace
        classifier_name: Best balanced classifier to use
        max_strategies: Limit analysis to N strategies (None = all)
        cost_config: Execution cost configuration
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    analyzer = StrategyAnalyzer(workspace_path)
    
    # Find all strategy files
    all_strategy_files = analyzer.find_strategy_files()
    
    if max_strategies:
        all_strategy_files = all_strategy_files[:max_strategies]
    
    print(f"Analyzing {len(all_strategy_files)} strategies across all types...")
    
    # Group strategies by type for organized analysis
    strategy_groups = {}
    for file_path in all_strategy_files:
        strategy_type = file_path.parent.name  # e.g., 'macd_crossover_grid'
        if strategy_type not in strategy_groups:
            strategy_groups[strategy_type] = []
        strategy_groups[strategy_type].append(file_path)
    
    print(f"Strategy types found: {list(strategy_groups.keys())}")
    for strategy_type, files in strategy_groups.items():
        print(f"  {strategy_type}: {len(files)} strategies")
    
    # Analyze all strategies
    return analyzer.analyze_multiple_strategies(
        all_strategy_files,
        classifier_name,
        cost_config
    )


def find_top_performers_per_regime(
    analysis_results: dict,
    regimes: list,
    top_n: int = 10,
    min_trades: int = 50
) -> dict:
    """
    Find top performing strategies for each regime.
    
    Args:
        analysis_results: Results from analyze_multiple_strategies
        regimes: List of regime names to analyze
        top_n: Number of top performers per regime
        min_trades: Minimum trades required to be considered
        
    Returns:
        Dictionary mapping regimes to top performer lists
    """
    regime_leaders = {}
    
    for regime in regimes:
        print(f"\nAnalyzing top performers in {regime.upper()} regime...")
        
        # Collect strategies with performance in this regime
        regime_performers = []
        
        for strategy_name, strategy_data in analysis_results['strategies'].items():
            regime_perf = strategy_data.get('regime_performance', {}).get(regime)
            
            if regime_perf and regime_perf['trade_count'] >= min_trades:
                performer_data = {
                    'strategy_name': strategy_name,
                    'strategy_type': strategy_data['strategy_file'].split('/')[-2],  # Extract type
                    'return': regime_perf['net_percentage_return'],
                    'log_return': regime_perf['total_net_log_return'],
                    'trades': regime_perf['trade_count'],
                    'win_rate': regime_perf['win_rate'],
                    'avg_trade_return': regime_perf['avg_trade_return'],
                    'best_trade': regime_perf['best_trade'],
                    'worst_trade': regime_perf['worst_trade'],
                    'profit_factor': regime_perf.get('profit_factor', 0),
                    'avg_bars_held': regime_perf['avg_bars_held']
                }
                regime_performers.append(performer_data)
        
        # Sort by return and take top N
        regime_performers.sort(key=lambda x: x['return'], reverse=True)
        top_performers = regime_performers[:top_n]
        
        regime_leaders[regime] = {
            'top_performers': top_performers,
            'total_qualified_strategies': len(regime_performers),
            'regime_stats': {
                'best_return': top_performers[0]['return'] if top_performers else 0,
                'worst_return': regime_performers[-1]['return'] if regime_performers else 0,
                'median_return': np.median([p['return'] for p in regime_performers]) if regime_performers else 0,
                'total_trades': sum(p['trades'] for p in regime_performers),
                'avg_win_rate': np.mean([p['win_rate'] for p in regime_performers]) if regime_performers else 0
            }
        }
        
        print(f"  Qualified strategies: {len(regime_performers)} (min {min_trades} trades)")
        print(f"  Best performer: {top_performers[0]['strategy_name'] if top_performers else 'None'}")
        print(f"    Return: {top_performers[0]['return']:.2%} ({top_performers[0]['trades']} trades)")
        
        if len(top_performers) > 1:
            print(f"  2nd best: {top_performers[1]['strategy_name']}")
            print(f"    Return: {top_performers[1]['return']:.2%} ({top_performers[1]['trades']} trades)")
    
    return regime_leaders


def analyze_regime_coverage(regime_leaders: dict) -> dict:
    """
    Analyze how well different strategy types cover different regimes.
    
    Args:
        regime_leaders: Results from find_top_performers_per_regime
        
    Returns:
        Dictionary with coverage analysis
    """
    print(f"\n{'='*80}")
    print("REGIME COVERAGE ANALYSIS")
    print("="*80)
    
    # Collect strategy types and their regime performance
    strategy_type_performance = {}
    
    for regime, regime_data in regime_leaders.items():
        for performer in regime_data['top_performers']:
            strategy_type = performer['strategy_type']
            
            if strategy_type not in strategy_type_performance:
                strategy_type_performance[strategy_type] = {}
            
            if regime not in strategy_type_performance[strategy_type]:
                strategy_type_performance[strategy_type][regime] = []
            
            strategy_type_performance[strategy_type][regime].append(performer)
    
    # Analyze coverage
    coverage_analysis = {}
    
    print(f"{'Strategy Type':<30} {'Regimes Covered':<20} {'Best Regime':<15} {'Best Return':<12}")
    print("-" * 80)
    
    for strategy_type, regime_data in strategy_type_performance.items():
        regimes_covered = list(regime_data.keys())
        
        # Find best performing regime for this strategy type
        best_regime = None
        best_return = float('-inf')
        
        for regime, performers in regime_data.items():
            top_performer = max(performers, key=lambda x: x['return'])
            if top_performer['return'] > best_return:
                best_return = top_performer['return']
                best_regime = regime
        
        coverage_analysis[strategy_type] = {
            'regimes_covered': regimes_covered,
            'coverage_count': len(regimes_covered),
            'best_regime': best_regime,
            'best_return': best_return,
            'avg_return_by_regime': {
                regime: np.mean([p['return'] for p in performers])
                for regime, performers in regime_data.items()
            }
        }
        
        print(f"{strategy_type:<30} {len(regimes_covered):<20} {best_regime:<15} {best_return:<12.2%}")
    
    return coverage_analysis


def build_ensemble_allocation(
    regime_leaders: dict,
    allocation_method: str = 'top_performer',
    diversification_limit: int = 3
) -> dict:
    """
    Build ensemble allocation based on regime performance.
    
    Args:
        regime_leaders: Results from find_top_performers_per_regime
        allocation_method: 'top_performer', 'top_n_equal', or 'return_weighted'
        diversification_limit: Maximum strategies per regime
        
    Returns:
        Dictionary with ensemble allocation
    """
    print(f"\n{'='*80}")
    print(f"BUILDING REGIME-ADAPTIVE ENSEMBLE")
    print(f"Allocation method: {allocation_method}")
    print(f"Diversification limit: {diversification_limit} strategies per regime")
    print("="*80)
    
    ensemble_allocation = {}
    
    for regime, regime_data in regime_leaders.items():
        top_performers = regime_data['top_performers']
        
        if not top_performers:
            print(f"{regime}: No qualified strategies")
            ensemble_allocation[regime] = []
            continue
        
        if allocation_method == 'top_performer':
            # Use only the best performer
            selected = [top_performers[0]]
            
        elif allocation_method == 'top_n_equal':
            # Use top N with equal weighting
            selected = top_performers[:diversification_limit]
            for strategy in selected:
                strategy['weight'] = 1.0 / len(selected)
                
        elif allocation_method == 'return_weighted':
            # Weight by relative returns (only positive returns)
            positive_performers = [p for p in top_performers[:diversification_limit] if p['return'] > 0]
            
            if positive_performers:
                total_return = sum(p['return'] for p in positive_performers)
                for strategy in positive_performers:
                    strategy['weight'] = strategy['return'] / total_return
                selected = positive_performers
            else:
                # Fall back to top performer if all negative
                selected = [top_performers[0]]
                selected[0]['weight'] = 1.0
        
        ensemble_allocation[regime] = selected
        
        print(f"\n{regime.upper()} REGIME ALLOCATION:")
        for i, strategy in enumerate(selected):
            weight = strategy.get('weight', 1.0)
            print(f"  {i+1}. {strategy['strategy_name']}")
            print(f"     Return: {strategy['return']:.2%}, Weight: {weight:.1%}")
            print(f"     Trades: {strategy['trades']}, Win Rate: {strategy['win_rate']:.1%}")
    
    return ensemble_allocation


def save_ensemble_results(
    regime_leaders: dict,
    coverage_analysis: dict,
    ensemble_allocation: dict,
    analysis_metadata: dict,
    output_dir: Path
):
    """Save all ensemble analysis results."""
    
    # Comprehensive results file
    results = {
        'analysis_metadata': analysis_metadata,
        'regime_leaders': regime_leaders,
        'coverage_analysis': coverage_analysis,
        'ensemble_allocation': ensemble_allocation,
        'summary_stats': {
            'total_regimes_analyzed': len(regime_leaders),
            'total_strategies_by_regime': {
                regime: data['total_qualified_strategies']
                for regime, data in regime_leaders.items()
            },
            'ensemble_strategies_selected': {
                regime: len(allocation)
                for regime, allocation in ensemble_allocation.items()
            }
        }
    }
    
    results_file = output_dir / "regime_adaptive_ensemble_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Ensemble configuration for implementation
    ensemble_config = {
        'classifier': analysis_metadata['classifier_name'],
        'regimes': list(ensemble_allocation.keys()),
        'allocation_method': 'regime_adaptive',
        'strategies_by_regime': {
            regime: [
                {
                    'strategy_name': s['strategy_name'],
                    'weight': s.get('weight', 1.0),
                    'expected_return': s['return'],
                    'trade_frequency': s['trades']
                }
                for s in strategies
            ]
            for regime, strategies in ensemble_allocation.items()
        },
        'implementation_notes': {
            'regime_detection': f"Use {analysis_metadata['classifier_name']} for real-time regime detection",
            'position_sizing': "Weight positions by regime allocation weights",
            'rebalancing': "Rebalance when regime changes detected",
            'minimum_trades': analysis_metadata.get('min_trades', 50)
        }
    }
    
    config_file = output_dir / "regime_adaptive_ensemble_config.json"
    with open(config_file, 'w') as f:
        json.dump(ensemble_config, f, indent=2, default=str)
    
    print(f"\n✅ Results saved:")
    print(f"  📊 Analysis: {results_file}")
    print(f"  ⚙️  Config: {config_file}")


def main():
    """Main ensemble building workflow."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    print("REGIME-ADAPTIVE ENSEMBLE BUILDER")
    print("="*50)
    print("Objective: Find top performers per regime for adaptive ensemble")
    print("="*50)
    
    # Use the best balanced classifier from previous analysis
    classifier_name = "SPY_market_regime_grid_0006_12"
    
    # Focus on main regimes (ignore rare trending regimes for now)
    main_regimes = ['bull_ranging', 'bear_ranging', 'neutral']
    
    # Execution cost configuration
    cost_config = ExecutionCostConfig(cost_multiplier=0.99)  # 1% cost
    
    print(f"Using classifier: {classifier_name}")
    print(f"Main regimes: {main_regimes}")
    print(f"Execution cost: 1% multiplier")
    
    # Step 1: Analyze full strategy universe (limit for initial analysis)
    print(f"\n{'='*60}")
    print("STEP 1: ANALYZING STRATEGY UNIVERSE")
    print("="*60)
    
    analysis_results = analyze_full_strategy_universe(
        workspace_path,
        classifier_name,
        max_strategies=50,  # Start with 50 for faster initial analysis
        cost_config=cost_config
    )
    
    if not analysis_results['strategies']:
        print("❌ No strategies analyzed successfully")
        return
    
    print(f"✅ Successfully analyzed {len(analysis_results['strategies'])} strategies")
    
    # Step 2: Find top performers per regime
    print(f"\n{'='*60}")
    print("STEP 2: FINDING TOP PERFORMERS PER REGIME")
    print("="*60)
    
    regime_leaders = find_top_performers_per_regime(
        analysis_results,
        main_regimes,
        top_n=10,
        min_trades=30  # Lower threshold to capture more strategies
    )
    
    # Step 3: Analyze regime coverage
    print(f"\n{'='*60}")
    print("STEP 3: ANALYZING REGIME COVERAGE")
    print("="*60)
    
    coverage_analysis = analyze_regime_coverage(regime_leaders)
    
    # Step 4: Build ensemble allocation
    print(f"\n{'='*60}")
    print("STEP 4: BUILDING ENSEMBLE ALLOCATION")
    print("="*60)
    
    ensemble_allocation = build_ensemble_allocation(
        regime_leaders,
        allocation_method='top_n_equal',  # Diversify with top 3 per regime
        diversification_limit=3
    )
    
    # Step 5: Save results
    analysis_metadata = {
        'classifier_name': classifier_name,
        'strategies_analyzed': len(analysis_results['strategies']),
        'cost_config': cost_config.__dict__,
        'regimes': main_regimes,
        'min_trades': 30,
        'analysis_date': pd.Timestamp.now().isoformat()
    }
    
    save_ensemble_results(
        regime_leaders,
        coverage_analysis,
        ensemble_allocation,
        analysis_metadata,
        workspace_path
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print("ENSEMBLE BUILDING COMPLETE")
    print("="*80)
    
    print("📈 REGIME PERFORMANCE SUMMARY:")
    for regime, regime_data in regime_leaders.items():
        if regime_data['top_performers']:
            best = regime_data['top_performers'][0]
            print(f"  {regime.upper()}: {best['strategy_name']} ({best['return']:.2%})")
        else:
            print(f"  {regime.upper()}: No qualified strategies")
    
    print(f"\n🎯 ENSEMBLE COMPOSITION:")
    total_strategies = sum(len(strategies) for strategies in ensemble_allocation.values())
    print(f"  Total strategies selected: {total_strategies}")
    
    for regime, strategies in ensemble_allocation.items():
        print(f"  {regime}: {len(strategies)} strategies")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"  1. Review ensemble configuration in generated files")
    print(f"  2. Implement regime detection and switching logic")  
    print(f"  3. Backtest the complete ensemble strategy")
    print(f"  4. Consider expanding analysis to full strategy universe")
    
    return ensemble_allocation


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n✅ Regime-adaptive ensemble successfully built!")
        
    except Exception as e:
        print(f"\n❌ Ensemble building failed: {e}")
        import traceback
        traceback.print_exc()