#!/usr/bin/env python3
"""
Build diversified ensemble with different strategy types per regime.

Enforces that each regime uses different strategy types to avoid correlation
and create true diversification.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_comprehensive_results():
    """Load comprehensive results."""
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    file_path = workspace_path / "comprehensive_regime_ensemble.json"
    
    with open(file_path, 'r') as f:
        return json.load(f)


def build_type_diversified_ensemble(results: dict) -> dict:
    """
    Build ensemble enforcing different strategy types per regime.
    
    Rules:
    1. No duplicate strategy types within a regime
    2. Select best performer of each type per regime
    3. 3 strategies per regime maximum
    """
    
    print("BUILDING TYPE-DIVERSIFIED REGIME ENSEMBLE")
    print("="*50)
    print("Rule: No duplicate strategy types within each regime")
    print("="*50)
    
    regime_champions = results['regime_champions']
    
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348,
        'neutral': 0.185
    }
    
    ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'type_diversified_regime_adaptive',
        'creation_date': pd.Timestamp.now().isoformat(),
        'diversification_rule': 'No duplicate strategy types within each regime',
        'regimes': {},
        'expected_performance': {}
    }
    
    total_expected_return = 0
    
    for regime in ['bull_ranging', 'bear_ranging', 'neutral']:
        print(f"\n🎯 {regime.upper()} REGIME - TYPE DIVERSIFICATION:")
        
        regime_data = regime_champions[regime]
        top_strategies = regime_data['top_10']
        
        # Group strategies by type and select best performer of each type
        strategy_types = defaultdict(list)
        for strategy in top_strategies:
            strategy_type = strategy['strategy_type']
            strategy_types[strategy_type].append(strategy)
        
        # Select best performer from each type
        type_representatives = []
        for strategy_type, strategies in strategy_types.items():
            # Sort by return and take best
            best_of_type = max(strategies, key=lambda x: x['return'])
            type_representatives.append(best_of_type)
        
        # Sort type representatives by return and take top 3
        type_representatives.sort(key=lambda x: x['return'], reverse=True)
        selected_strategies = type_representatives[:3]
        
        # Show what we're selecting vs avoiding
        print(f"  Available strategy types: {len(strategy_types)}")
        for strategy_type, strategies in sorted(strategy_types.items(), 
                                              key=lambda x: max(s['return'] for s in x[1]), 
                                              reverse=True):
            best_return = max(s['return'] for s in strategies)
            is_selected = any(s['strategy_type'] == strategy_type for s in selected_strategies)
            status = "✅ SELECTED" if is_selected else "   available"
            print(f"    {status} {strategy_type}: {len(strategies)} strategies (best: {best_return:.2%})")
        
        # Weight by performance
        total_return = sum(s['return'] for s in selected_strategies if s['return'] > 0)
        
        weighted_strategies = []
        regime_expected_return = 0
        
        print(f"\n  Selected strategies:")
        for i, strategy in enumerate(selected_strategies):
            if strategy['return'] > 0 and total_return > 0:
                weight = strategy['return'] / total_return
            else:
                weight = 1.0 / len(selected_strategies)
            
            weighted_strategies.append({
                'strategy_name': strategy['strategy_name'],
                'strategy_type': strategy['strategy_type'],
                'expected_return': strategy['return'],
                'weight': weight,
                'trades': strategy['trades'],
                'win_rate': strategy['win_rate'],
                'profit_factor': strategy['profit_factor'],
                'rank': i + 1
            })
            
            regime_expected_return += strategy['return'] * weight
            
            print(f"    {i+1}. {strategy['strategy_name']} ({strategy['strategy_type']})")
            print(f"       Return: {strategy['return']:.2%} | Weight: {weight:.1%} | Trades: {strategy['trades']}")
        
        ensemble['regimes'][regime] = {
            'strategies': weighted_strategies,
            'regime_expected_return': regime_expected_return,
            'regime_frequency': regime_frequencies[regime],
            'type_diversity': len(set(s['strategy_type'] for s in weighted_strategies))
        }
        
        regime_contribution = regime_expected_return * regime_frequencies[regime]
        total_expected_return += regime_contribution
        
        print(f"  📊 Regime expected return: {regime_expected_return:.2%}")
        print(f"  📊 Contribution to total: {regime_contribution:.2%}")
        print(f"  🎯 Type diversity: {len(set(s['strategy_type'] for s in weighted_strategies))}/3 unique types")
    
    ensemble['expected_performance'] = {
        'total_expected_return': total_expected_return,
        'regime_contributions': {
            regime: data['regime_expected_return'] * data['regime_frequency']
            for regime, data in ensemble['regimes'].items()
        }
    }
    
    print(f"\n🎯 TYPE-DIVERSIFIED ENSEMBLE EXPECTED RETURN: {total_expected_return:.2%}")
    
    return ensemble


def analyze_correlation_reduction(results: dict, type_diversified_ensemble: dict):
    """Analyze how type diversification reduces correlation risk."""
    
    print(f"\n{'='*80}")
    print("CORRELATION RISK REDUCTION ANALYSIS")
    print("="*80)
    
    # Compare original vs type-diversified approach
    regime_champions = results['regime_champions']
    
    for regime in ['bull_ranging', 'bear_ranging', 'neutral']:
        print(f"\n📊 {regime.upper()} REGIME COMPARISON:")
        
        # Original approach (top 3 regardless of type)
        original_top3 = regime_champions[regime]['top_10'][:3]
        original_types = [s['strategy_type'] for s in original_top3]
        original_unique_types = len(set(original_types))
        original_return = sum(s['return'] for s in original_top3) / 3  # Simple average
        
        print(f"  Original approach:")
        for i, strategy in enumerate(original_top3):
            print(f"    {i+1}. {strategy['strategy_name']} ({strategy['strategy_type']}) - {strategy['return']:.2%}")
        print(f"    Unique types: {original_unique_types}/3")
        print(f"    Average return: {original_return:.2%}")
        
        # Type-diversified approach
        diversified_strategies = type_diversified_ensemble['regimes'][regime]['strategies']
        diversified_types = [s['strategy_type'] for s in diversified_strategies]
        diversified_unique_types = len(set(diversified_types))
        diversified_return = type_diversified_ensemble['regimes'][regime]['regime_expected_return']
        
        print(f"  Type-diversified approach:")
        for i, strategy in enumerate(diversified_strategies):
            print(f"    {i+1}. {strategy['strategy_name']} ({strategy['strategy_type']}) - {strategy['expected_return']:.2%}")
        print(f"    Unique types: {diversified_unique_types}/3")
        print(f"    Weighted return: {diversified_return:.2%}")
        
        # Risk assessment
        correlation_risk = "HIGH" if original_unique_types < 3 else "LOW"
        diversification_improvement = diversified_unique_types - original_unique_types
        return_cost = original_return - diversified_return
        
        print(f"  📊 Analysis:")
        print(f"    Original correlation risk: {correlation_risk}")
        print(f"    Diversification improvement: +{diversification_improvement} unique types")
        print(f"    Return trade-off: {return_cost:+.2%}")


def build_cross_regime_enhanced_ensemble(results: dict, type_diversified_ensemble: dict) -> dict:
    """
    Build final ensemble: Type-diversified specialists + cross-regime performers.
    """
    
    print(f"\n{'='*80}")
    print("BUILDING ENHANCED ENSEMBLE: TYPE-DIVERSIFIED + CROSS-REGIME")
    print("="*80)
    
    # Cross-regime performers (top performers across multiple regimes)
    cross_regime_performers = [
        {'name': 'SPY_dema_crossover_grid_19_15', 'type': 'dema_crossover_grid', 'regimes': 3, 'weighted_return': 0.1308},
        {'name': 'SPY_elder_ray_grid_13_0_-0.001', 'type': 'elder_ray_grid', 'regimes': 2, 'weighted_return': 0.0899},
        {'name': 'SPY_sma_crossover_grid_19_15', 'type': 'sma_crossover_grid', 'regimes': 2, 'weighted_return': 0.0868},
    ]
    
    # 75% type-diversified specialists + 25% cross-regime performers
    specialist_weight = 0.75
    cross_regime_weight = 0.25
    
    specialist_return = type_diversified_ensemble['expected_performance']['total_expected_return']
    
    # Cross-regime component return (average of top 3)
    cross_regime_return = sum(p['weighted_return'] for p in cross_regime_performers) / len(cross_regime_performers)
    
    # Final ensemble return
    final_return = (specialist_return * specialist_weight) + (cross_regime_return * cross_regime_weight)
    
    enhanced_ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'type_diversified_with_cross_regime_hedge',
        'creation_date': pd.Timestamp.now().isoformat(),
        'allocation': {
            'regime_specialists': specialist_weight,
            'cross_regime_hedge': cross_regime_weight
        },
        'regime_specialists': type_diversified_ensemble['regimes'],
        'cross_regime_performers': [
            {
                'strategy_name': p['name'],
                'strategy_type': p['type'],
                'weighted_return': p['weighted_return'],
                'regimes_covered': p['regimes'],
                'weight': 1.0 / len(cross_regime_performers)
            }
            for p in cross_regime_performers
        ],
        'expected_performance': {
            'total_expected_return': final_return,
            'specialist_contribution': specialist_return * specialist_weight,
            'cross_regime_contribution': cross_regime_return * cross_regime_weight
        },
        'risk_benefits': {
            'type_diversification': 'Each regime uses different strategy types',
            'regime_hedge': '25% allocation hedges regime misclassification',
            'correlation_reduction': 'Minimized correlation within and across regimes'
        }
    }
    
    print(f"Allocation:")
    print(f"  Type-diversified specialists (75%): {specialist_return:.2%}")
    print(f"  Cross-regime hedge (25%): {cross_regime_return:.2%}")
    print(f"  🎯 Enhanced Ensemble Return: {final_return:.2%}")
    
    print(f"\nCross-regime performers:")
    for performer in cross_regime_performers:
        print(f"  • {performer['name']} ({performer['type']})")
        print(f"    Return: {performer['weighted_return']:.2%} | Regimes: {performer['regimes']}")
    
    return enhanced_ensemble


def main():
    """Main workflow."""
    
    print("TYPE-DIVERSIFIED ENSEMBLE BUILDER")
    print("="*40)
    print("Goal: Eliminate correlation risk from duplicate strategy types")
    print("="*40)
    
    # Load comprehensive results
    results = load_comprehensive_results()
    
    # Build type-diversified ensemble
    type_diversified_ensemble = build_type_diversified_ensemble(results)
    
    # Analyze correlation reduction
    analyze_correlation_reduction(results, type_diversified_ensemble)
    
    # Build enhanced ensemble with cross-regime hedge
    enhanced_ensemble = build_cross_regime_enhanced_ensemble(results, type_diversified_ensemble)
    
    # Save results
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    # Save type-diversified ensemble
    type_div_file = workspace_path / "type_diversified_ensemble.json"
    with open(type_div_file, 'w') as f:
        json.dump({
            'type_diversified_ensemble': type_diversified_ensemble,
            'methodology': 'Enforce different strategy types within each regime'
        }, f, indent=2, default=str)
    
    # Save enhanced ensemble
    enhanced_file = workspace_path / "enhanced_type_diversified_ensemble.json"
    with open(enhanced_file, 'w') as f:
        json.dump(enhanced_ensemble, f, indent=2, default=str)
    
    # Create implementation config
    impl_config = {
        'ensemble_name': 'enhanced_type_diversified_regime_adaptive',
        'classifier': enhanced_ensemble['classifier'],
        'allocation': enhanced_ensemble['allocation'],
        'regime_strategies': {
            regime: [
                {
                    'strategy_name': s['strategy_name'],
                    'strategy_type': s['strategy_type'],
                    'weight': s['weight'],
                    'expected_return': s['expected_return']
                }
                for s in regime_data['strategies']
            ]
            for regime, regime_data in enhanced_ensemble['regime_specialists'].items()
        },
        'cross_regime_strategies': [
            {
                'strategy_name': s['strategy_name'],
                'strategy_type': s['strategy_type'],
                'weight': s['weight'],
                'weighted_return': s['weighted_return']
            }
            for s in enhanced_ensemble['cross_regime_performers']
        ],
        'expected_performance': enhanced_ensemble['expected_performance']
    }
    
    impl_config_file = workspace_path / "enhanced_type_diversified_config.json"
    with open(impl_config_file, 'w') as f:
        json.dump(impl_config, f, indent=2, default=str)
    
    print(f"\n✅ TYPE-DIVERSIFIED ENSEMBLE COMPLETE!")
    print(f"  📊 Type-diversified: {type_div_file}")
    print(f"  🎯 Enhanced ensemble: {enhanced_file}")
    print(f"  ⚙️  Implementation config: {impl_config_file}")
    print(f"  💰 Expected return: {enhanced_ensemble['expected_performance']['total_expected_return']:.2%}")
    
    print(f"\n🎯 KEY BENEFITS:")
    print(f"  ✅ No duplicate strategy types within regimes")
    print(f"  ✅ True diversification across strategy approaches")
    print(f"  ✅ 25% cross-regime hedge for stability")
    print(f"  ✅ Reduced correlation risk")
    
    return enhanced_ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 TYPE-DIVERSIFIED ENSEMBLE READY!")
        
    except Exception as e:
        print(f"\n❌ Type-diversified ensemble building failed: {e}")
        import traceback
        traceback.print_exc()