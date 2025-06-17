#!/usr/bin/env python3
"""
Comprehensive analysis of ALL 1,229 strategies for regime-adaptive ensemble.

This script analyzes every single strategy to find:
1. True best performers (not just samples)
2. Real cross-regime performers
3. Drawdown characteristics and risk metrics
4. Optimal ensemble with risk-adjusted performance
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.sparse_trace_analysis import (
    StrategyAnalyzer,
    ExecutionCostConfig
)


def get_all_strategy_files(workspace_path: Path) -> List[Path]:
    """Get ALL strategy files (not just samples)."""
    
    signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
    all_files = []
    
    for strategy_type_dir in signals_dir.iterdir():
        if strategy_type_dir.is_dir():
            strategy_files = list(strategy_type_dir.glob("*.parquet"))
            all_files.extend(strategy_files)
    
    return sorted(all_files)


def analyze_strategy_batch_parallel(
    strategy_files: List[Path],
    classifier_name: str,
    workspace_path: Path,
    cost_config: ExecutionCostConfig,
    batch_id: int,
    batch_size: int = 50
):
    """Analyze strategies in parallel batches."""
    
    print(f"Batch {batch_id}: Processing {len(strategy_files)} strategies...")
    start_time = time.time()
    
    try:
        analyzer = StrategyAnalyzer(workspace_path)
        results = analyzer.analyze_multiple_strategies(
            strategy_files,
            classifier_name,
            cost_config
        )
        
        elapsed = time.time() - start_time
        success_count = len(results.get('strategies', {}))
        
        print(f"  ✅ Batch {batch_id}: {success_count}/{len(strategy_files)} strategies analyzed in {elapsed:.1f}s")
        
        return batch_id, results
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ Batch {batch_id} failed after {elapsed:.1f}s: {e}")
        return batch_id, None


def find_true_cross_regime_performers(all_results: dict, main_regimes: list) -> Dict[str, dict]:
    """
    Find strategies that actually perform well across multiple regimes.
    
    Unlike the previous analysis, this looks at actual performance data
    across regimes, not just top-N rankings.
    """
    
    print(f"\n{'='*80}")
    print("FINDING TRUE CROSS-REGIME PERFORMERS")
    print("="*80)
    
    # Collect performance data for all strategies across all regimes
    strategy_cross_performance = {}
    
    for strategy_name, strategy_data in all_results.get('strategies', {}).items():
        regime_performance = strategy_data.get('regime_performance', {})
        
        # Only consider strategies with meaningful performance in multiple regimes
        qualifying_regimes = []
        regime_returns = {}
        
        for regime in main_regimes:
            regime_perf = regime_performance.get(regime)
            if regime_perf and regime_perf['trade_count'] >= 30 and regime_perf['net_percentage_return'] > 0:
                qualifying_regimes.append(regime)
                regime_returns[regime] = regime_perf['net_percentage_return']
        
        # Must perform positively in at least 2 regimes
        if len(qualifying_regimes) >= 2:
            strategy_type = strategy_data['strategy_file'].split('/')[-2]
            
            # Calculate cross-regime metrics
            returns = list(regime_returns.values())
            avg_return = np.mean(returns)
            min_return = min(returns)
            return_consistency = min_return / max(returns)  # Consistency score
            
            # Weight by regime frequencies
            regime_frequencies = {'bull_ranging': 0.447, 'bear_ranging': 0.348, 'neutral': 0.185}
            weighted_return = sum(
                regime_returns[regime] * regime_frequencies.get(regime, 0)
                for regime in regime_returns
            )
            
            strategy_cross_performance[strategy_name] = {
                'strategy_type': strategy_type,
                'qualifying_regimes': qualifying_regimes,
                'regime_count': len(qualifying_regimes),
                'regime_returns': regime_returns,
                'avg_return': avg_return,
                'min_return': min_return,
                'return_consistency': return_consistency,
                'weighted_return': weighted_return,
                'regime_performance_details': {
                    regime: regime_performance[regime] 
                    for regime in qualifying_regimes
                }
            }
    
    # Sort by weighted return
    sorted_cross_performers = sorted(
        strategy_cross_performance.items(),
        key=lambda x: x[1]['weighted_return'],
        reverse=True
    )
    
    print(f"Found {len(strategy_cross_performance)} strategies with positive performance in 2+ regimes:")
    print(f"{'Strategy':<50} {'Type':<25} {'Regimes':<10} {'Weighted Return':<15} {'Consistency':<12}")
    print("-" * 115)
    
    for strategy_name, perf_data in sorted_cross_performers[:20]:  # Show top 20
        regimes_str = ','.join(perf_data['qualifying_regimes'])
        print(f"{strategy_name:<50} {perf_data['strategy_type']:<25} {perf_data['regime_count']:<10} "
              f"{perf_data['weighted_return']:<15.2%} {perf_data['return_consistency']:<12.2f}")
    
    return dict(sorted_cross_performers)


def calculate_drawdown_metrics(all_results: dict) -> Dict[str, dict]:
    """
    Calculate drawdown and risk metrics for strategies.
    
    Since we don't have equity curves, we'll estimate risk metrics
    from available trade data.
    """
    
    print(f"\n{'='*80}")
    print("CALCULATING RISK AND DRAWDOWN METRICS")
    print("="*80)
    
    strategy_risk_metrics = {}
    
    for strategy_name, strategy_data in all_results.get('strategies', {}).items():
        overall_perf = strategy_data.get('overall_performance', {})
        
        if not overall_perf:
            continue
        
        # Available metrics
        total_return = overall_perf.get('net_percentage_return', 0)
        total_trades = overall_perf.get('trade_count', 0)
        win_rate = overall_perf.get('win_rate', 0)
        avg_trade_return = overall_perf.get('avg_trade_return', 0)
        best_trade = overall_perf.get('best_trade', 0)
        worst_trade = overall_perf.get('worst_trade', 0)
        profit_factor = overall_perf.get('profit_factor', 0)
        
        if total_trades < 30:  # Skip strategies with too few trades
            continue
        
        # Estimate risk metrics
        # This is simplified since we don't have full equity curves
        trade_volatility = abs(worst_trade - best_trade) if best_trade and worst_trade else 0
        
        # Estimate Sharpe ratio (simplified)
        if trade_volatility > 0:
            estimated_sharpe = (avg_trade_return * np.sqrt(total_trades)) / trade_volatility
        else:
            estimated_sharpe = 0
        
        # Risk-adjusted return (return per unit of worst drawdown)
        if worst_trade < 0:
            risk_adjusted_return = total_return / abs(worst_trade)
        else:
            risk_adjusted_return = total_return
        
        strategy_risk_metrics[strategy_name] = {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'worst_trade': worst_trade,
            'best_trade': best_trade,
            'trade_volatility': trade_volatility,
            'estimated_sharpe': estimated_sharpe,
            'risk_adjusted_return': risk_adjusted_return,
            'strategy_type': strategy_data['strategy_file'].split('/')[-2]
        }
    
    # Sort by risk-adjusted return
    sorted_by_risk_adj = sorted(
        strategy_risk_metrics.items(),
        key=lambda x: x[1]['risk_adjusted_return'],
        reverse=True
    )
    
    print(f"Risk metrics calculated for {len(strategy_risk_metrics)} strategies")
    print(f"{'Strategy':<50} {'Type':<25} {'Return':<10} {'Risk-Adj':<10} {'Sharpe':<8}")
    print("-" * 105)
    
    for strategy_name, metrics in sorted_by_risk_adj[:15]:
        print(f"{strategy_name:<50} {metrics['strategy_type']:<25} "
              f"{metrics['total_return']:<10.2%} {metrics['risk_adjusted_return']:<10.2f} "
              f"{metrics['estimated_sharpe']:<8.2f}")
    
    return dict(sorted_by_risk_adj)


def build_risk_optimized_ensemble(
    regime_champions: dict,
    cross_regime_performers: dict,
    risk_metrics: dict,
    main_regimes: list
) -> dict:
    """
    Build ensemble optimized for risk-adjusted returns and drawdown reduction.
    """
    
    print(f"\n{'='*80}")
    print("BUILDING RISK-OPTIMIZED ENSEMBLE")
    print("="*80)
    
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348,
        'neutral': 0.185
    }
    
    ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'risk_optimized_regime_adaptive',
        'creation_date': pd.Timestamp.now().isoformat(),
        'allocation_strategy': {
            'regime_specialists': 0.70,  # 70% to regime specialists
            'cross_regime_performers': 0.30  # 30% to cross-regime performers
        },
        'regimes': {},
        'cross_regime_allocation': {},
        'expected_performance': {}
    }
    
    # 1. Select top regime specialists with good risk metrics
    print("\n🎯 REGIME SPECIALISTS (70% allocation):")
    
    total_specialist_return = 0
    
    for regime in main_regimes:
        print(f"\n{regime.upper()} REGIME:")
        
        # Get candidates for this regime
        regime_candidates = []
        
        for strategy_name, strategy_data in regime_champions[regime]['top_10']:
            # Get risk metrics for this strategy
            risk_data = risk_metrics.get(strategy_name)
            if risk_data:
                candidate = {
                    'strategy_name': strategy_name,
                    'strategy_type': strategy_data['strategy_type'],
                    'regime_return': strategy_data['return'],
                    'risk_adjusted_return': risk_data['risk_adjusted_return'],
                    'estimated_sharpe': risk_data['estimated_sharpe'],
                    'worst_trade': risk_data['worst_trade'],
                    'profit_factor': strategy_data['profit_factor'],
                    'trades': strategy_data['trades']
                }
                regime_candidates.append(candidate)
        
        # Sort by risk-adjusted return and select top 2
        regime_candidates.sort(key=lambda x: x['risk_adjusted_return'], reverse=True)
        selected_specialists = regime_candidates[:2]
        
        # Weight by risk-adjusted performance
        total_risk_adj = sum(s['risk_adjusted_return'] for s in selected_specialists if s['risk_adjusted_return'] > 0)
        
        for specialist in selected_specialists:
            if total_risk_adj > 0:
                specialist['weight'] = specialist['risk_adjusted_return'] / total_risk_adj
            else:
                specialist['weight'] = 0.5
        
        # Normalize weights
        total_weight = sum(s['weight'] for s in selected_specialists)
        if total_weight > 0:
            for specialist in selected_specialists:
                specialist['weight'] = specialist['weight'] / total_weight
        
        regime_expected_return = sum(
            s['regime_return'] * s['weight'] for s in selected_specialists
        )
        
        ensemble['regimes'][regime] = {
            'specialists': selected_specialists,
            'regime_expected_return': regime_expected_return,
            'regime_frequency': regime_frequencies[regime]
        }
        
        regime_contribution = regime_expected_return * regime_frequencies[regime] * 0.70
        total_specialist_return += regime_contribution
        
        for i, specialist in enumerate(selected_specialists):
            print(f"  {i+1}. {specialist['strategy_name']:<45}")
            print(f"     Return: {specialist['regime_return']:>8.2%} | "
                  f"Risk-Adj: {specialist['risk_adjusted_return']:>6.2f} | "
                  f"Weight: {specialist['weight']:>6.1%}")
        
        print(f"  📊 Regime expected return: {regime_expected_return:.2%}")
    
    # 2. Select cross-regime performers
    print(f"\n🌐 CROSS-REGIME PERFORMERS (30% allocation):")
    
    cross_regime_return = 0
    
    if cross_regime_performers:
        # Take top 3 cross-regime performers with good risk metrics
        top_cross_regime = []
        
        for strategy_name, perf_data in list(cross_regime_performers.items())[:10]:
            risk_data = risk_metrics.get(strategy_name)
            if risk_data:
                candidate = {
                    'strategy_name': strategy_name,
                    'strategy_type': perf_data['strategy_type'],
                    'weighted_return': perf_data['weighted_return'],
                    'regime_count': perf_data['regime_count'],
                    'return_consistency': perf_data['return_consistency'],
                    'risk_adjusted_return': risk_data['risk_adjusted_return'],
                    'estimated_sharpe': risk_data['estimated_sharpe'],
                    'qualifying_regimes': perf_data['qualifying_regimes']
                }
                top_cross_regime.append(candidate)
        
        # Sort by combination of consistency and risk-adjusted return
        top_cross_regime.sort(key=lambda x: x['return_consistency'] * x['risk_adjusted_return'], reverse=True)
        selected_cross_regime = top_cross_regime[:3]
        
        # Weight by weighted return
        total_weighted_return = sum(s['weighted_return'] for s in selected_cross_regime if s['weighted_return'] > 0)
        
        for performer in selected_cross_regime:
            if total_weighted_return > 0:
                performer['weight'] = performer['weighted_return'] / total_weighted_return
            else:
                performer['weight'] = 1.0 / len(selected_cross_regime)
        
        cross_regime_return = sum(
            p['weighted_return'] * p['weight'] for p in selected_cross_regime
        ) * 0.30
        
        ensemble['cross_regime_allocation'] = {
            'performers': selected_cross_regime,
            'expected_return': cross_regime_return / 0.30,  # Actual return before weighting
            'allocation_weight': 0.30
        }
        
        for i, performer in enumerate(selected_cross_regime):
            print(f"  {i+1}. {performer['strategy_name']:<45}")
            print(f"     Weighted Return: {performer['weighted_return']:>8.2%} | "
                  f"Consistency: {performer['return_consistency']:>6.2f} | "
                  f"Weight: {performer['weight']:>6.1%}")
            print(f"     Regimes: {', '.join(performer['qualifying_regimes'])}")
    
    # 3. Calculate total expected performance
    total_expected_return = total_specialist_return + cross_regime_return
    
    ensemble['expected_performance'] = {
        'total_expected_return': total_expected_return,
        'specialist_contribution': total_specialist_return,
        'cross_regime_contribution': cross_regime_return,
        'risk_characteristics': {
            'diversification': 'High - Multiple strategies per regime + cross-regime hedge',
            'regime_misclassification_protection': 'Strong - Cross-regime performers buffer regime errors',
            'drawdown_protection': 'Enhanced - Risk-adjusted strategy selection'
        }
    }
    
    print(f"\n🎯 RISK-OPTIMIZED ENSEMBLE EXPECTED RETURN: {total_expected_return:.2%}")
    print(f"   Specialist contribution: {total_specialist_return:.2%}")
    print(f"   Cross-regime contribution: {cross_regime_return:.2%}")
    
    return ensemble


def main():
    """Main comprehensive analysis workflow."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    print("COMPREHENSIVE ALL-STRATEGIES ANALYSIS")
    print("="*60)
    print("Analyzing ALL 1,229 strategies for optimal regime-adaptive ensemble")
    print("="*60)
    
    # Configuration
    classifier_name = "SPY_market_regime_grid_0006_12"
    cost_config = ExecutionCostConfig(cost_multiplier=0.99)
    main_regimes = ['bull_ranging', 'bear_ranging', 'neutral']
    
    # Get ALL strategy files
    print("Step 1: Collecting all strategy files...")
    all_strategy_files = get_all_strategy_files(workspace_path)
    print(f"Found {len(all_strategy_files)} total strategies to analyze")
    
    # Analyze in batches for memory management
    print(f"\nStep 2: Analyzing all strategies in batches...")
    batch_size = 100  # Larger batches for efficiency
    batches = [all_strategy_files[i:i+batch_size] for i in range(0, len(all_strategy_files), batch_size)]
    
    print(f"Processing {len(batches)} batches of ~{batch_size} strategies each")
    
    all_batch_results = {}
    analyzer = StrategyAnalyzer(workspace_path)
    
    for i, batch_files in enumerate(batches, 1):
        print(f"\nProcessing batch {i}/{len(batches)} ({len(batch_files)} strategies)...")
        
        try:
            batch_results = analyzer.analyze_multiple_strategies(
                batch_files,
                classifier_name,
                cost_config
            )
            
            # Merge results
            if 'strategies' not in all_batch_results:
                all_batch_results = batch_results
            else:
                all_batch_results['strategies'].update(batch_results.get('strategies', {}))
            
            success_count = len(batch_results.get('strategies', {}))
            print(f"  ✅ Batch {i}: {success_count}/{len(batch_files)} strategies analyzed")
            
        except Exception as e:
            print(f"  ❌ Batch {i} failed: {e}")
    
    total_analyzed = len(all_batch_results.get('strategies', {}))
    print(f"\n✅ Total strategies analyzed: {total_analyzed}/{len(all_strategy_files)}")
    
    if total_analyzed == 0:
        print("❌ No strategies analyzed successfully. Exiting.")
        return
    
    # Find regime champions from full dataset
    print(f"\nStep 3: Finding regime champions from ALL strategies...")
    
    regime_champions = {}
    for regime in main_regimes:
        print(f"Analyzing {regime} regime...")
        
        regime_performers = []
        for strategy_name, strategy_data in all_batch_results['strategies'].items():
            regime_perf = strategy_data.get('regime_performance', {}).get(regime)
            
            if regime_perf and regime_perf['trade_count'] >= 30:
                performer = {
                    'strategy_name': strategy_name,
                    'strategy_type': strategy_data['strategy_file'].split('/')[-2],
                    'return': regime_perf['net_percentage_return'],
                    'trades': regime_perf['trade_count'],
                    'win_rate': regime_perf['win_rate'],
                    'profit_factor': regime_perf.get('profit_factor', 0),
                    'avg_trade_return': regime_perf['avg_trade_return']
                }
                regime_performers.append(performer)
        
        # Sort by return and take top 10
        regime_performers.sort(key=lambda x: x['return'], reverse=True)
        
        regime_champions[regime] = {
            'top_10': regime_performers[:10],
            'total_candidates': len(regime_performers)
        }
        
        print(f"  {regime}: {len(regime_performers)} qualified strategies")
        if regime_performers:
            print(f"    Best: {regime_performers[0]['strategy_name']} ({regime_performers[0]['return']:.2%})")
    
    # Find true cross-regime performers
    cross_regime_performers = find_true_cross_regime_performers(all_batch_results, main_regimes)
    
    # Calculate risk metrics
    risk_metrics = calculate_drawdown_metrics(all_batch_results)
    
    # Build risk-optimized ensemble
    risk_optimized_ensemble = build_risk_optimized_ensemble(
        regime_champions, cross_regime_performers, risk_metrics, main_regimes
    )
    
    # Save comprehensive results
    output_file = workspace_path / "comprehensive_all_strategies_analysis.json"
    
    final_results = {
        'analysis_metadata': {
            'total_strategies_available': len(all_strategy_files),
            'total_strategies_analyzed': total_analyzed,
            'classifier_name': classifier_name,
            'cost_config': cost_config.__dict__,
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'regime_champions_full': regime_champions,
        'cross_regime_performers': cross_regime_performers,
        'risk_metrics': risk_metrics,
        'risk_optimized_ensemble': risk_optimized_ensemble
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Save implementation config
    impl_config_file = workspace_path / "risk_optimized_ensemble_config.json"
    
    impl_config = {
        'ensemble_name': 'risk_optimized_regime_adaptive',
        'classifier': risk_optimized_ensemble['classifier'],
        'allocation_strategy': risk_optimized_ensemble['allocation_strategy'],
        'regime_strategies': {
            regime: [
                {
                    'strategy_name': s['strategy_name'],
                    'weight': s['weight'],
                    'expected_return': s['regime_return'],
                    'risk_score': s['risk_adjusted_return']
                }
                for s in regime_data['specialists']
            ]
            for regime, regime_data in risk_optimized_ensemble['regimes'].items()
        },
        'cross_regime_strategies': [
            {
                'strategy_name': p['strategy_name'],
                'weight': p['weight'],
                'weighted_return': p['weighted_return'],
                'qualifying_regimes': p['qualifying_regimes']
            }
            for p in risk_optimized_ensemble.get('cross_regime_allocation', {}).get('performers', [])
        ],
        'expected_performance': risk_optimized_ensemble['expected_performance']
    }
    
    with open(impl_config_file, 'w') as f:
        json.dump(impl_config, f, indent=2, default=str)
    
    print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"  📊 Full results: {output_file}")
    print(f"  ⚙️  Implementation config: {impl_config_file}")
    print(f"  🎯 Risk-optimized expected return: {risk_optimized_ensemble['expected_performance']['total_expected_return']:.2%}")
    
    # Show top cross-regime performers
    if cross_regime_performers:
        print(f"\n🌟 TOP CROSS-REGIME PERFORMERS FOUND:")
        for strategy_name, perf_data in list(cross_regime_performers.items())[:5]:
            print(f"  • {strategy_name}")
            print(f"    Regimes: {', '.join(perf_data['qualifying_regimes'])}")
            print(f"    Weighted return: {perf_data['weighted_return']:.2%}")
            print(f"    Consistency: {perf_data['return_consistency']:.2f}")
    
    return risk_optimized_ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 COMPREHENSIVE RISK-OPTIMIZED ENSEMBLE READY!")
        
    except Exception as e:
        print(f"\n❌ Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()