#!/usr/bin/env python3
"""
Build diversified regime-adaptive ensemble with multiple strategies per regime
and cross-regime performers for stability.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_comprehensive_results(file_path: Path) -> dict:
    """Load the comprehensive ensemble results."""
    with open(file_path, 'r') as f:
        return json.load(f)


def find_cross_regime_performers(regime_champions: dict, min_regimes: int = 2) -> Dict[str, dict]:
    """
    Find strategies that perform well across multiple regimes.
    
    Args:
        regime_champions: Results from comprehensive analysis
        min_regimes: Minimum number of regimes a strategy must appear in top performers
        
    Returns:
        Dictionary of cross-regime performers with their performance across regimes
    """
    
    print(f"\n{'='*80}")
    print("IDENTIFYING CROSS-REGIME PERFORMERS")
    print("="*80)
    
    # Collect all strategy names and their regime performances
    strategy_performance = {}
    
    for regime, regime_data in regime_champions.items():
        if not regime_data:
            continue
            
        # Look at top 10 performers (not just top 5)
        top_performers = regime_data.get('top_10', regime_data.get('top_5', []))
        
        for performer in top_performers:
            strategy_name = performer['strategy_name']
            
            if strategy_name not in strategy_performance:
                strategy_performance[strategy_name] = {
                    'strategy_type': performer['strategy_type'],
                    'regimes': {},
                    'total_regimes': 0,
                    'avg_return': 0,
                    'weighted_return': 0
                }
            
            strategy_performance[strategy_name]['regimes'][regime] = {
                'return': performer['return'],
                'trades': performer['trades'],
                'win_rate': performer['win_rate'],
                'profit_factor': performer['profit_factor'],
                'rank': top_performers.index(performer) + 1
            }
    
    # Calculate statistics for each strategy
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348,
        'neutral': 0.185
    }
    
    cross_regime_performers = {}
    
    for strategy_name, perf_data in strategy_performance.items():
        regime_count = len(perf_data['regimes'])
        
        if regime_count >= min_regimes:
            # Calculate average and weighted returns
            returns = [regime_perf['return'] for regime_perf in perf_data['regimes'].values()]
            avg_return = np.mean(returns)
            
            # Weight by regime frequency
            weighted_return = sum(
                regime_perf['return'] * regime_frequencies.get(regime, 0)
                for regime, regime_perf in perf_data['regimes'].items()
            )
            
            perf_data['total_regimes'] = regime_count
            perf_data['avg_return'] = avg_return
            perf_data['weighted_return'] = weighted_return
            
            cross_regime_performers[strategy_name] = perf_data
    
    # Sort by weighted return
    sorted_performers = sorted(
        cross_regime_performers.items(),
        key=lambda x: x[1]['weighted_return'],
        reverse=True
    )
    
    print(f"Found {len(cross_regime_performers)} strategies appearing in {min_regimes}+ regimes:")
    print(f"{'Strategy':<45} {'Type':<25} {'Regimes':<10} {'Avg Return':<12} {'Weighted Return':<15}")
    print("-" * 110)
    
    for strategy_name, perf_data in sorted_performers[:10]:  # Show top 10
        regimes_str = ','.join(perf_data['regimes'].keys())
        print(f"{strategy_name:<45} {perf_data['strategy_type']:<25} {perf_data['total_regimes']:<10} "
              f"{perf_data['avg_return']:<12.2%} {perf_data['weighted_return']:<15.2%}")
    
    return dict(sorted_performers)


def build_diversified_ensemble(
    regime_champions: dict,
    cross_regime_performers: dict,
    strategies_per_regime: int = 3,
    cross_regime_allocation: float = 0.2
) -> dict:
    """
    Build diversified ensemble with multiple strategies per regime and cross-regime performers.
    
    Args:
        regime_champions: Top performers per regime
        cross_regime_performers: Strategies that work across regimes
        strategies_per_regime: Number of specialist strategies per regime
        cross_regime_allocation: Fraction of allocation to cross-regime performers (0.0-1.0)
        
    Returns:
        Diversified ensemble configuration
    """
    
    print(f"\n{'='*80}")
    print("BUILDING DIVERSIFIED REGIME-ADAPTIVE ENSEMBLE")
    print(f"Strategies per regime: {strategies_per_regime}")
    print(f"Cross-regime allocation: {cross_regime_allocation:.1%}")
    print("="*80)
    
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348,
        'neutral': 0.185
    }
    
    diversified_ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'diversified_regime_adaptive',
        'creation_date': pd.Timestamp.now().isoformat(),
        'diversification_config': {
            'strategies_per_regime': strategies_per_regime,
            'cross_regime_allocation': cross_regime_allocation,
            'regime_specialist_allocation': 1 - cross_regime_allocation
        },
        'regimes': {},
        'cross_regime_strategies': {},
        'expected_performance': {}
    }
    
    total_expected_return = 0
    
    # 1. Add regime specialists
    print("\n🎯 REGIME SPECIALISTS:")
    specialist_allocation = 1 - cross_regime_allocation
    
    for regime, regime_data in regime_champions.items():
        if not regime_data:
            continue
            
        print(f"\n{regime.upper()} REGIME SPECIALISTS:")
        
        # Take top N strategies for this regime
        top_strategies = regime_data.get('top_10', regime_data.get('top_5', []))[:strategies_per_regime]
        
        # Weight strategies within regime (higher weight for better performers)
        total_regime_return = sum(s['return'] for s in top_strategies if s['return'] > 0)
        
        regime_strategies = []
        regime_expected_return = 0
        
        for i, strategy in enumerate(top_strategies):
            if strategy['return'] > 0:
                # Weight by relative performance within regime
                strategy_weight = strategy['return'] / total_regime_return if total_regime_return > 0 else 1/len(top_strategies)
            else:
                strategy_weight = 0.1 / len(top_strategies)  # Small weight for negative performers
            
            regime_strategies.append({
                'strategy_name': strategy['strategy_name'],
                'strategy_type': strategy['strategy_type'],
                'expected_return': strategy['return'],
                'weight_in_regime': strategy_weight,
                'trades': strategy['trades'],
                'win_rate': strategy['win_rate'],
                'profit_factor': strategy['profit_factor'],
                'rank': i + 1
            })
            
            regime_expected_return += strategy['return'] * strategy_weight
            
            print(f"  {i+1}. {strategy['strategy_name']:<40} "
                  f"Return: {strategy['return']:>8.2%} Weight: {strategy_weight:>6.1%}")
        
        diversified_ensemble['regimes'][regime] = {
            'strategies': regime_strategies,
            'regime_expected_return': regime_expected_return,
            'regime_frequency': regime_frequencies[regime],
            'allocation_weight': specialist_allocation
        }
        
        # Contribution to total return
        regime_contribution = regime_expected_return * regime_frequencies[regime] * specialist_allocation
        total_expected_return += regime_contribution
        
        print(f"  📊 Regime expected return: {regime_expected_return:.2%}")
        print(f"  📊 Contribution to total: {regime_contribution:.2%}")
    
    # 2. Add cross-regime performers
    print(f"\n🌐 CROSS-REGIME PERFORMERS ({cross_regime_allocation:.1%} allocation):")
    
    if cross_regime_allocation > 0 and cross_regime_performers:
        # Take top cross-regime performers
        top_cross_regime = list(cross_regime_performers.items())[:3]  # Top 3 cross-regime
        
        if top_cross_regime:
            # Weight cross-regime strategies by their weighted returns
            total_cross_return = sum(perf_data['weighted_return'] for _, perf_data in top_cross_regime if perf_data['weighted_return'] > 0)
            
            cross_regime_strategies = []
            cross_regime_expected_return = 0
            
            for strategy_name, perf_data in top_cross_regime:
                if perf_data['weighted_return'] > 0:
                    strategy_weight = perf_data['weighted_return'] / total_cross_return if total_cross_return > 0 else 1/len(top_cross_regime)
                else:
                    strategy_weight = 0.1 / len(top_cross_regime)
                
                cross_regime_strategies.append({
                    'strategy_name': strategy_name,
                    'strategy_type': perf_data['strategy_type'],
                    'weighted_return': perf_data['weighted_return'],
                    'avg_return': perf_data['avg_return'],
                    'weight': strategy_weight,
                    'regimes_count': perf_data['total_regimes'],
                    'regime_performance': perf_data['regimes']
                })
                
                cross_regime_expected_return += perf_data['weighted_return'] * strategy_weight
                
                print(f"  • {strategy_name:<40} "
                      f"Weighted Return: {perf_data['weighted_return']:>8.2%} "
                      f"Weight: {strategy_weight:>6.1%} "
                      f"({perf_data['total_regimes']} regimes)")
            
            diversified_ensemble['cross_regime_strategies'] = {
                'strategies': cross_regime_strategies,
                'expected_return': cross_regime_expected_return,
                'allocation_weight': cross_regime_allocation
            }
            
            # Add cross-regime contribution
            cross_regime_contribution = cross_regime_expected_return * cross_regime_allocation
            total_expected_return += cross_regime_contribution
            
            print(f"  📊 Cross-regime expected return: {cross_regime_expected_return:.2%}")
            print(f"  📊 Contribution to total: {cross_regime_contribution:.2%}")
        else:
            print("  No qualified cross-regime performers found")
    
    # 3. Calculate final expected performance
    diversified_ensemble['expected_performance'] = {
        'total_expected_return': total_expected_return,
        'regime_specialist_contribution': total_expected_return - (cross_regime_expected_return * cross_regime_allocation if 'cross_regime_expected_return' in locals() else 0),
        'cross_regime_contribution': cross_regime_expected_return * cross_regime_allocation if 'cross_regime_expected_return' in locals() else 0,
        'breakdown': {
            regime: {
                'contribution': data['regime_expected_return'] * data['regime_frequency'] * data['allocation_weight']
            }
            for regime, data in diversified_ensemble['regimes'].items()
        }
    }
    
    print(f"\n🎯 DIVERSIFIED ENSEMBLE EXPECTED RETURN: {total_expected_return:.2%}")
    
    return diversified_ensemble


def compare_ensemble_approaches(simple_ensemble: dict, diversified_ensemble: dict):
    """Compare simple vs diversified ensemble approaches."""
    
    print(f"\n{'='*80}")
    print("ENSEMBLE APPROACH COMPARISON")
    print("="*80)
    
    simple_return = simple_ensemble['expected_performance']['total_expected_return']
    diversified_return = diversified_ensemble['expected_performance']['total_expected_return']
    
    print(f"📊 EXPECTED RETURNS:")
    print(f"  Simple (1 strategy per regime):      {simple_return:.2%}")
    print(f"  Diversified (multiple + cross):      {diversified_return:.2%}")
    print(f"  Difference:                          {diversified_return - simple_return:+.2%}")
    
    print(f"\n📊 RISK CHARACTERISTICS:")
    print(f"  Simple ensemble:")
    print(f"    • 3 total strategies")
    print(f"    • Single point of failure per regime")
    print(f"    • High regime classification dependency")
    
    print(f"  Diversified ensemble:")
    diversified_strategy_count = sum(len(regime_data['strategies']) for regime_data in diversified_ensemble['regimes'].values())
    if 'cross_regime_strategies' in diversified_ensemble and diversified_ensemble['cross_regime_strategies']:
        diversified_strategy_count += len(diversified_ensemble['cross_regime_strategies']['strategies'])
    
    print(f"    • {diversified_strategy_count} total strategies")
    print(f"    • Multiple strategies per regime for robustness")
    print(f"    • Cross-regime performers as regime misclassification hedge")
    print(f"    • Lower concentration risk")
    
    return {
        'simple_return': simple_return,
        'diversified_return': diversified_return,
        'return_difference': diversified_return - simple_return,
        'diversification_benefit': diversified_return > simple_return
    }


def main():
    """Main diversified ensemble building workflow."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    results_file = workspace_path / "comprehensive_regime_ensemble.json"
    
    print("DIVERSIFIED REGIME-ADAPTIVE ENSEMBLE BUILDER")
    print("="*60)
    print("Building robust ensemble with multiple strategies per regime")
    print("and cross-regime performers for stability")
    print("="*60)
    
    # Load comprehensive results
    results = load_comprehensive_results(results_file)
    regime_champions = results['regime_champions']
    simple_ensemble = results['optimal_ensemble']
    
    # Find cross-regime performers
    cross_regime_performers = find_cross_regime_performers(regime_champions, min_regimes=2)
    
    # Build diversified ensemble
    diversified_ensemble = build_diversified_ensemble(
        regime_champions,
        cross_regime_performers,
        strategies_per_regime=3,      # 3 strategies per regime
        cross_regime_allocation=0.15  # 15% to cross-regime performers
    )
    
    # Compare approaches
    comparison = compare_ensemble_approaches(simple_ensemble, diversified_ensemble)
    
    # Save diversified ensemble
    diversified_file = workspace_path / "diversified_regime_ensemble.json"
    
    final_results = {
        'diversified_ensemble': diversified_ensemble,
        'cross_regime_performers': cross_regime_performers,
        'ensemble_comparison': comparison,
        'analysis_metadata': {
            'creation_date': pd.Timestamp.now().isoformat(),
            'base_analysis': str(results_file),
            'diversification_approach': 'multi_strategy_per_regime_plus_cross_regime'
        }
    }
    
    with open(diversified_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Save implementation config
    implementation_config = {
        'classifier': diversified_ensemble['classifier'],
        'ensemble_type': 'diversified_regime_adaptive',
        'regime_strategies': {
            regime: [
                {
                    'strategy_name': s['strategy_name'],
                    'weight': s['weight_in_regime'],
                    'expected_return': s['expected_return']
                }
                for s in regime_data['strategies']
            ]
            for regime, regime_data in diversified_ensemble['regimes'].items()
        },
        'cross_regime_strategies': [
            {
                'strategy_name': s['strategy_name'],
                'weight': s['weight'],
                'weighted_return': s['weighted_return']
            }
            for s in diversified_ensemble.get('cross_regime_strategies', {}).get('strategies', [])
        ] if diversified_ensemble.get('cross_regime_strategies') else [],
        'allocation_framework': {
            'regime_specialist_weight': diversified_ensemble['diversification_config']['regime_specialist_allocation'],
            'cross_regime_weight': diversified_ensemble['diversification_config']['cross_regime_allocation']
        },
        'expected_return': diversified_ensemble['expected_performance']['total_expected_return']
    }
    
    config_file = workspace_path / "diversified_ensemble_config.json"
    with open(config_file, 'w') as f:
        json.dump(implementation_config, f, indent=2, default=str)
    
    print(f"\n✅ DIVERSIFIED ENSEMBLE COMPLETE!")
    print(f"  📊 Full analysis: {diversified_file}")
    print(f"  ⚙️  Implementation config: {config_file}")
    
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print(f"  Expected return: {diversified_ensemble['expected_performance']['total_expected_return']:.2%}")
    print(f"  Risk reduction: Multiple strategies per regime + cross-regime hedge")
    print(f"  Implementation: Use regime detection + weighted allocation")
    
    return diversified_ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 DIVERSIFIED REGIME-ADAPTIVE ENSEMBLE READY!")
        
    except Exception as e:
        print(f"\n❌ Diversified ensemble building failed: {e}")
        import traceback
        traceback.print_exc()