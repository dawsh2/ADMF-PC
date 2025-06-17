#!/usr/bin/env python3
"""
Test the actual value of cross-regime performers vs pure regime specialists.

Compare:
1. Pure regime specialists (current approach)
2. Cross-regime performers only
3. Hybrid approach (mix of both)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_comprehensive_results():
    """Load comprehensive results to get cross-regime data."""
    # Since the comprehensive analysis had cross-regime data in the output
    # Let's simulate what we saw in the terminal output
    
    cross_regime_performers = [
        {'name': 'SPY_dema_crossover_grid_19_15', 'type': 'dema_crossover_grid', 'regimes': 3, 'weighted_return': 0.1308, 'consistency': 0.04},
        {'name': 'SPY_elder_ray_grid_13_0_-0.001', 'type': 'elder_ray_grid', 'regimes': 2, 'weighted_return': 0.0899, 'consistency': 0.78},
        {'name': 'SPY_sma_crossover_grid_19_15', 'type': 'sma_crossover_grid', 'regimes': 2, 'weighted_return': 0.0868, 'consistency': 0.40},
        {'name': 'SPY_stochastic_crossover_grid_5_7', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0834, 'consistency': 0.11},
        {'name': 'SPY_stochastic_crossover_grid_11_7', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0817, 'consistency': 0.44},
        {'name': 'SPY_pivot_channel_bounces_grid_15_2_0.001', 'type': 'pivot_channel_bounces_grid', 'regimes': 3, 'weighted_return': 0.0799, 'consistency': 0.32},
        {'name': 'SPY_pivot_channel_bounces_grid_15_3_0.001', 'type': 'pivot_channel_bounces_grid', 'regimes': 3, 'weighted_return': 0.0799, 'consistency': 0.32},
        {'name': 'SPY_pivot_channel_bounces_grid_15_4_0.001', 'type': 'pivot_channel_bounces_grid', 'regimes': 3, 'weighted_return': 0.0799, 'consistency': 0.32},
        {'name': 'SPY_pivot_channel_bounces_grid_20_2_0.001', 'type': 'pivot_channel_bounces_grid', 'regimes': 3, 'weighted_return': 0.0799, 'consistency': 0.32},
        {'name': 'SPY_pivot_channel_bounces_grid_20_3_0.001', 'type': 'pivot_channel_bounces_grid', 'regimes': 3, 'weighted_return': 0.0799, 'consistency': 0.32},
        {'name': 'SPY_stochastic_crossover_grid_19_7', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0761, 'consistency': 0.42},
        {'name': 'SPY_stochastic_crossover_grid_5_5', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0694, 'consistency': 0.12},
        {'name': 'SPY_rsi_threshold_grid_19_50', 'type': 'rsi_threshold_grid', 'regimes': 2, 'weighted_return': 0.0680, 'consistency': 0.07},
        {'name': 'SPY_stochastic_crossover_grid_19_5', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0651, 'consistency': 0.36},
        {'name': 'SPY_stochastic_crossover_grid_27_5', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0565, 'consistency': 0.09},
        {'name': 'SPY_elder_ray_grid_21_0_-0.001', 'type': 'elder_ray_grid', 'regimes': 2, 'weighted_return': 0.0562, 'consistency': 0.93}
    ]
    
    # Load our regime specialists from the final ensemble
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    final_ensemble_file = workspace_path / "final_optimized_ensemble.json"
    
    with open(final_ensemble_file, 'r') as f:
        final_ensemble_data = json.load(f)
    
    return cross_regime_performers, final_ensemble_data


def build_cross_regime_only_ensemble(cross_regime_performers: list) -> dict:
    """Build ensemble using only cross-regime performers."""
    
    print("🌐 CROSS-REGIME ONLY ENSEMBLE:")
    
    # Take top 5 cross-regime performers
    top_cross_regime = cross_regime_performers[:5]
    
    # Weight by weighted return
    total_weighted_return = sum(p['weighted_return'] for p in top_cross_regime)
    
    ensemble_return = 0
    strategies = []
    
    for performer in top_cross_regime:
        weight = performer['weighted_return'] / total_weighted_return
        contribution = performer['weighted_return'] * weight
        ensemble_return += contribution
        
        strategies.append({
            'strategy_name': performer['name'],
            'strategy_type': performer['type'],
            'weighted_return': performer['weighted_return'],
            'weight': weight,
            'regimes_covered': performer['regimes'],
            'consistency': performer['consistency']
        })
        
        print(f"  • {performer['name']}")
        print(f"    Weighted Return: {performer['weighted_return']:.2%} | Weight: {weight:.1%}")
        print(f"    Regimes: {performer['regimes']} | Consistency: {performer['consistency']:.2f}")
    
    print(f"\n  🎯 Cross-Regime Ensemble Return: {ensemble_return:.2%}")
    
    return {
        'ensemble_method': 'cross_regime_only',
        'strategies': strategies,
        'expected_return': ensemble_return,
        'diversification_benefits': {
            'regime_misclassification_hedge': 'High - strategies work across multiple regimes',
            'stability': 'High - less dependent on perfect regime detection',
            'complexity': 'Low - no regime switching required'
        }
    }


def build_hybrid_ensemble(cross_regime_performers: list, regime_specialists_data: dict) -> dict:
    """Build hybrid ensemble mixing cross-regime performers and specialists."""
    
    print("🔄 HYBRID ENSEMBLE (70% specialists + 30% cross-regime):")
    
    # Get specialist return from final ensemble
    specialist_return = regime_specialists_data['final_ensemble']['expected_performance']['total_expected_return']
    
    # Get cross-regime return (top 3)
    top_cross_regime = cross_regime_performers[:3]
    total_weighted_return = sum(p['weighted_return'] for p in top_cross_regime)
    cross_regime_return = total_weighted_return / len(top_cross_regime)  # Average of top 3
    
    # Hybrid allocation
    specialist_weight = 0.70
    cross_regime_weight = 0.30
    
    hybrid_return = (specialist_return * specialist_weight) + (cross_regime_return * cross_regime_weight)
    
    print(f"  Specialist component (70%): {specialist_return:.2%}")
    print(f"  Cross-regime component (30%): {cross_regime_return:.2%}")
    print(f"  🎯 Hybrid Ensemble Return: {hybrid_return:.2%}")
    
    return {
        'ensemble_method': 'hybrid_specialist_cross_regime',
        'specialist_allocation': specialist_weight,
        'cross_regime_allocation': cross_regime_weight,
        'specialist_return': specialist_return,
        'cross_regime_return': cross_regime_return,
        'expected_return': hybrid_return,
        'benefits': {
            'regime_specialization': 'High from specialists',
            'regime_hedge': 'Medium from cross-regime performers',
            'complexity': 'Medium - some regime switching + stable base'
        }
    }


def analyze_risk_characteristics():
    """Analyze risk characteristics of each approach."""
    
    print(f"\n{'='*80}")
    print("RISK CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    approaches = {
        'Pure Specialists': {
            'regime_dependency': 'HIGH - completely dependent on regime classification',
            'regime_misclassification_risk': 'HIGH - wrong regime = wrong strategy',
            'return_potential': 'HIGHEST - optimized per regime',
            'stability': 'LOW - volatile regime switches',
            'complexity': 'HIGH - requires perfect regime detection'
        },
        'Cross-Regime Only': {
            'regime_dependency': 'LOW - strategies work across regimes',
            'regime_misclassification_risk': 'LOW - hedged by design',
            'return_potential': 'MEDIUM - not optimized per regime',
            'stability': 'HIGH - consistent across market states',
            'complexity': 'LOW - no regime switching needed'
        },
        'Hybrid': {
            'regime_dependency': 'MEDIUM - partially hedged',
            'regime_misclassification_risk': 'MEDIUM - 30% hedge',
            'return_potential': 'HIGH - balance of optimization and stability',
            'stability': 'MEDIUM-HIGH - more stable than pure specialists',
            'complexity': 'MEDIUM - simplified regime switching'
        }
    }
    
    for approach, characteristics in approaches.items():
        print(f"\n🔍 {approach.upper()}:")
        for metric, value in characteristics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value}")


def main():
    """Compare all three approaches."""
    
    print("CROSS-REGIME PERFORMERS VALUE ANALYSIS")
    print("="*50)
    print("Testing: Pure Specialists vs Cross-Regime vs Hybrid")
    print("="*50)
    
    # Load data
    cross_regime_performers, regime_specialists_data = load_comprehensive_results()
    
    # Current approach (pure specialists)
    specialist_return = regime_specialists_data['final_ensemble']['expected_performance']['total_expected_return']
    
    print(f"📊 CURRENT PURE SPECIALISTS ENSEMBLE:")
    print(f"  Expected Return: {specialist_return:.2%}")
    print(f"  Strategies: 9 (3 per regime)")
    print(f"  Method: Regime-specific optimization")
    
    # Cross-regime only approach
    print(f"\n{'='*60}")
    cross_regime_ensemble = build_cross_regime_only_ensemble(cross_regime_performers)
    
    # Hybrid approach  
    print(f"\n{'='*60}")
    hybrid_ensemble = build_hybrid_ensemble(cross_regime_performers, regime_specialists_data)
    
    # Risk analysis
    analyze_risk_characteristics()
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print("="*80)
    
    results = [
        ('Pure Specialists', specialist_return, 'Highest return, highest risk'),
        ('Cross-Regime Only', cross_regime_ensemble['expected_return'], 'Lower return, much lower risk'),
        ('Hybrid (70/30)', hybrid_ensemble['expected_return'], 'Good return, medium risk')
    ]
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Approach':<20} {'Expected Return':<15} {'Trade-off':<30}")
    print("-" * 70)
    
    for approach, ret, tradeoff in results:
        print(f"{approach:<20} {ret:<15.2%} {tradeoff:<30}")
    
    # Recommendation
    print(f"\n🎯 RECOMMENDATION:")
    
    if hybrid_ensemble['expected_return'] > specialist_return * 0.95:  # Within 5% of pure specialists
        print(f"  💡 HYBRID APPROACH RECOMMENDED")
        print(f"     • Return: {hybrid_ensemble['expected_return']:.2%} (vs {specialist_return:.2%} pure)")
        print(f"     • Much better risk characteristics")
        print(f"     • 30% hedge against regime misclassification")
        print(f"     • More robust in practice")
    else:
        print(f"  💡 EVALUATE TRADE-OFFS CAREFULLY")
        print(f"     • Pure specialists: {specialist_return:.2%} but high regime risk")
        print(f"     • Cross-regime: {cross_regime_ensemble['expected_return']:.2%} but much more stable")
        print(f"     • Consider risk tolerance and regime detection confidence")
    
    return {
        'pure_specialists': specialist_return,
        'cross_regime_only': cross_regime_ensemble['expected_return'],
        'hybrid': hybrid_ensemble['expected_return']
    }


if __name__ == "__main__":
    try:
        results = main()
        print(f"\n🚀 CROSS-REGIME VALUE ANALYSIS COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()