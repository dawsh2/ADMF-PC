#!/usr/bin/env python3
"""
Build two-layer ensemble architecture:
1. Baseline Layer: Always-active cross-regime performers
2. Regime Layer: Regime-specific boosters

This provides stability from baseline + optimization from regime specialists.
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


def select_baseline_cross_regime_performers() -> list:
    """
    Select baseline always-active strategies that performed well across regimes.
    
    These run continuously regardless of regime detection.
    """
    
    print("🌐 SELECTING BASELINE CROSS-REGIME PERFORMERS")
    print("="*60)
    print("Criteria: Positive performance in multiple regimes, different strategy types")
    
    # Cross-regime performers from our comprehensive analysis
    cross_regime_candidates = [
        {'name': 'SPY_dema_crossover_grid_19_15', 'type': 'dema_crossover_grid', 'regimes': 3, 'weighted_return': 0.1308, 'consistency': 0.04},
        {'name': 'SPY_elder_ray_grid_13_0_-0.001', 'type': 'elder_ray_grid', 'regimes': 2, 'weighted_return': 0.0899, 'consistency': 0.78},
        {'name': 'SPY_sma_crossover_grid_19_15', 'type': 'sma_crossover_grid', 'regimes': 2, 'weighted_return': 0.0868, 'consistency': 0.40},
        {'name': 'SPY_stochastic_crossover_grid_5_7', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0834, 'consistency': 0.11},
        {'name': 'SPY_stochastic_crossover_grid_11_7', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0817, 'consistency': 0.44},
        {'name': 'SPY_pivot_channel_bounces_grid_15_2_0.001', 'type': 'pivot_channel_bounces_grid', 'regimes': 3, 'weighted_return': 0.0799, 'consistency': 0.32},
        {'name': 'SPY_stochastic_crossover_grid_19_7', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0761, 'consistency': 0.42},
        {'name': 'SPY_rsi_threshold_grid_19_50', 'type': 'rsi_threshold_grid', 'regimes': 2, 'weighted_return': 0.0680, 'consistency': 0.07},
        {'name': 'SPY_stochastic_crossover_grid_19_5', 'type': 'stochastic_crossover_grid', 'regimes': 2, 'weighted_return': 0.0651, 'consistency': 0.36},
        {'name': 'SPY_tema_sma_crossover_grid_19_15', 'type': 'tema_sma_crossover_grid', 'regimes': 2, 'weighted_return': 0.0620, 'consistency': 0.51},
        {'name': 'SPY_ema_crossover_grid_19_15', 'type': 'ema_crossover_grid', 'regimes': 2, 'weighted_return': 0.0585, 'consistency': 0.44},
        {'name': 'SPY_williams_r_grid_21_-80_-20', 'type': 'williams_r_grid', 'regimes': 2, 'weighted_return': 0.0570, 'consistency': 0.65}
    ]
    
    # Select top performers with different strategy types (max 5)
    selected_baseline = []
    used_types = set()
    
    # Sort by weighted return and select best of each type
    cross_regime_candidates.sort(key=lambda x: x['weighted_return'], reverse=True)
    
    for candidate in cross_regime_candidates:
        if candidate['type'] not in used_types and len(selected_baseline) < 5:
            selected_baseline.append(candidate)
            used_types.add(candidate['type'])
    
    print(f"Selected {len(selected_baseline)} baseline strategies:")
    
    total_baseline_return = 0
    for i, strategy in enumerate(selected_baseline):
        weight = 1.0 / len(selected_baseline)  # Equal weighting for baseline
        total_baseline_return += strategy['weighted_return'] * weight
        
        print(f"  {i+1}. {strategy['name']} ({strategy['type']})")
        print(f"     Weighted Return: {strategy['weighted_return']:.2%} | "
              f"Regimes: {strategy['regimes']} | "
              f"Consistency: {strategy['consistency']:.2f}")
    
    print(f"\n📊 Baseline Layer Expected Return: {total_baseline_return:.2%}")
    
    return selected_baseline, total_baseline_return


def select_regime_boosters(results: dict, baseline_types: set) -> dict:
    """
    Select regime-specific boosters, avoiding baseline strategy types.
    
    These activate only when their specific regime is detected.
    """
    
    print(f"\n🎯 SELECTING REGIME-SPECIFIC BOOSTERS")
    print("="*60)
    print(f"Avoiding baseline types: {baseline_types}")
    
    regime_champions = results['regime_champions']
    regime_boosters = {}
    
    for regime in ['bull_ranging', 'bear_ranging', 'neutral']:
        print(f"\n{regime.upper()} REGIME BOOSTERS:")
        
        top_strategies = regime_champions[regime]['top_10']
        
        # Group by strategy type, excluding baseline types
        strategy_types = defaultdict(list)
        for strategy in top_strategies:
            strategy_type = strategy['strategy_type']
            if strategy_type not in baseline_types:  # Avoid baseline types
                strategy_types[strategy_type].append(strategy)
        
        # Select best performer from each type (max 4 boosters per regime)
        selected_boosters = []
        for strategy_type, strategies in strategy_types.items():
            if len(selected_boosters) >= 4:  # Max 4 boosters per regime
                break
            best_of_type = max(strategies, key=lambda x: x['return'])
            selected_boosters.append(best_of_type)
        
        # Sort by return and take top performers
        selected_boosters.sort(key=lambda x: x['return'], reverse=True)
        selected_boosters = selected_boosters[:4]  # Top 4 boosters
        
        # Weight by performance
        total_return = sum(s['return'] for s in selected_boosters if s['return'] > 0)
        
        weighted_boosters = []
        regime_booster_return = 0
        
        print(f"  Available strategy types (excluding baseline): {len(strategy_types)}")
        print(f"  Selected boosters:")
        
        for i, strategy in enumerate(selected_boosters):
            if strategy['return'] > 0 and total_return > 0:
                weight = strategy['return'] / total_return
            else:
                weight = 1.0 / len(selected_boosters)
            
            weighted_boosters.append({
                'strategy_name': strategy['strategy_name'],
                'strategy_type': strategy['strategy_type'],
                'expected_return': strategy['return'],
                'weight': weight,
                'trades': strategy['trades'],
                'win_rate': strategy['win_rate'],
                'profit_factor': strategy['profit_factor']
            })
            
            regime_booster_return += strategy['return'] * weight
            
            print(f"    {i+1}. {strategy['strategy_name']} ({strategy['strategy_type']})")
            print(f"       Return: {strategy['return']:.2%} | Weight: {weight:.1%}")
        
        regime_boosters[regime] = {
            'boosters': weighted_boosters,
            'regime_booster_return': regime_booster_return,
            'regime_frequency': {'bull_ranging': 0.447, 'bear_ranging': 0.348, 'neutral': 0.185}[regime]
        }
        
        print(f"  📊 {regime.title()} Booster Return: {regime_booster_return:.2%}")
    
    return regime_boosters


def build_two_layer_ensemble(
    baseline_strategies: list,
    baseline_return: float,
    regime_boosters: dict,
    baseline_allocation: float = 0.6
) -> dict:
    """
    Build final two-layer ensemble.
    
    Args:
        baseline_allocation: Fraction allocated to always-active baseline (0.0-1.0)
    """
    
    print(f"\n{'='*80}")
    print("BUILDING TWO-LAYER ENSEMBLE")
    print(f"Baseline allocation: {baseline_allocation:.0%} | Regime allocation: {1-baseline_allocation:.0%}")
    print("="*80)
    
    regime_allocation = 1 - baseline_allocation
    
    # Calculate regime booster contribution
    regime_booster_contribution = 0
    for regime, booster_data in regime_boosters.items():
        regime_contrib = (booster_data['regime_booster_return'] * 
                         booster_data['regime_frequency'] * 
                         regime_allocation)
        regime_booster_contribution += regime_contrib
    
    # Total ensemble return
    total_ensemble_return = (baseline_return * baseline_allocation) + regime_booster_contribution
    
    ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'two_layer_baseline_plus_regime_boosters',
        'creation_date': pd.Timestamp.now().isoformat(),
        'architecture': {
            'baseline_layer': {
                'description': 'Always-active cross-regime performers',
                'allocation': baseline_allocation,
                'expected_return': baseline_return,
                'contribution': baseline_return * baseline_allocation
            },
            'regime_layer': {
                'description': 'Regime-specific boosters (activate per regime)',
                'allocation': regime_allocation,
                'expected_return': regime_booster_contribution / regime_allocation,
                'contribution': regime_booster_contribution
            }
        },
        'baseline_strategies': [
            {
                'strategy_name': s['name'],
                'strategy_type': s['type'],
                'weighted_return': s['weighted_return'],
                'regimes_covered': s['regimes'],
                'weight': 1.0 / len(baseline_strategies),
                'status': 'always_active'
            }
            for s in baseline_strategies
        ],
        'regime_boosters': {
            regime: {
                'boosters': booster_data['boosters'],
                'regime_expected_return': booster_data['regime_booster_return'],
                'regime_frequency': booster_data['regime_frequency'],
                'status': f'active_when_{regime}_detected'
            }
            for regime, booster_data in regime_boosters.items()
        },
        'expected_performance': {
            'total_expected_return': total_ensemble_return,
            'baseline_contribution': baseline_return * baseline_allocation,
            'regime_booster_contribution': regime_booster_contribution
        }
    }
    
    print(f"\n📊 TWO-LAYER ENSEMBLE PERFORMANCE:")
    print(f"  Baseline Layer ({baseline_allocation:.0%}): {baseline_return:.2%} → {baseline_return * baseline_allocation:.2%} contribution")
    print(f"  Regime Layer ({regime_allocation:.0%}): {regime_booster_contribution / regime_allocation:.2%} → {regime_booster_contribution:.2%} contribution")
    print(f"  🎯 Total Expected Return: {total_ensemble_return:.2%}")
    
    print(f"\n🔄 REGIME BOOSTER BREAKDOWN:")
    for regime, booster_data in regime_boosters.items():
        regime_contrib = (booster_data['regime_booster_return'] * 
                         booster_data['regime_frequency'] * 
                         regime_allocation)
        print(f"  {regime.title()}: {booster_data['regime_booster_return']:.2%} × {booster_data['regime_frequency']:.1%} × {regime_allocation:.0%} = {regime_contrib:.2%}")
    
    return ensemble


def create_implementation_config(ensemble: dict) -> dict:
    """Create implementation-ready configuration."""
    
    return {
        'ensemble_name': 'two_layer_baseline_plus_regime_boosters',
        'classifier': ensemble['classifier'],
        'architecture': ensemble['architecture'],
        'implementation': {
            'baseline_strategies': {
                'description': 'Always active regardless of regime',
                'allocation_method': 'equal_weight',
                'strategies': ensemble['baseline_strategies']
            },
            'regime_boosters': {
                'description': 'Activate only when specific regime detected',
                'allocation_method': 'performance_weighted_within_regime',
                'boosters_by_regime': {
                    regime: [
                        {
                            'strategy_name': b['strategy_name'],
                            'strategy_type': b['strategy_type'],
                            'weight': b['weight'],
                            'expected_return': b['expected_return']
                        }
                        for b in regime_data['boosters']
                    ]
                    for regime, regime_data in ensemble['regime_boosters'].items()
                }
            }
        },
        'expected_performance': ensemble['expected_performance'],
        'operational_notes': {
            'baseline_execution': 'Run baseline strategies continuously',
            'regime_detection': f"Use {ensemble['classifier']} for regime classification",
            'regime_boosters': 'Activate/deactivate boosters based on regime changes',
            'rebalancing': 'Rebalance when regime changes or periodically'
        }
    }


def main():
    """Main two-layer ensemble building workflow."""
    
    print("TWO-LAYER ENSEMBLE BUILDER")
    print("="*40)
    print("Architecture: Always-active baseline + regime-specific boosters")
    print("="*40)
    
    # Load comprehensive results
    results = load_comprehensive_results()
    
    # Step 1: Select baseline cross-regime performers
    baseline_strategies, baseline_return = select_baseline_cross_regime_performers()
    baseline_types = {s['type'] for s in baseline_strategies}
    
    # Step 2: Select regime boosters (avoiding baseline types)
    regime_boosters = select_regime_boosters(results, baseline_types)
    
    # Step 3: Build two-layer ensemble (60% baseline, 40% regime boosters)
    two_layer_ensemble = build_two_layer_ensemble(
        baseline_strategies, baseline_return, regime_boosters, 
        baseline_allocation=0.6
    )
    
    # Step 4: Create implementation config
    impl_config = create_implementation_config(two_layer_ensemble)
    
    # Save results
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    # Save two-layer ensemble
    ensemble_file = workspace_path / "two_layer_ensemble.json"
    with open(ensemble_file, 'w') as f:
        json.dump(two_layer_ensemble, f, indent=2, default=str)
    
    # Save implementation config
    config_file = workspace_path / "two_layer_ensemble_config.json"
    with open(config_file, 'w') as f:
        json.dump(impl_config, f, indent=2, default=str)
    
    print(f"\n✅ TWO-LAYER ENSEMBLE COMPLETE!")
    print(f"  📊 Ensemble: {ensemble_file}")
    print(f"  ⚙️  Config: {config_file}")
    print(f"  💰 Expected return: {two_layer_ensemble['expected_performance']['total_expected_return']:.2%}")
    
    # Summary
    print(f"\n📋 ENSEMBLE SUMMARY:")
    print(f"  🌐 Baseline strategies: {len(baseline_strategies)} (always active)")
    print(f"  🎯 Regime boosters: {sum(len(r['boosters']) for r in regime_boosters.values())} total")
    for regime, booster_data in regime_boosters.items():
        print(f"    • {regime}: {len(booster_data['boosters'])} boosters")
    print(f"  ⚖️  Allocation: 60% baseline + 40% regime boosters")
    print(f"  🔧 No strategy type overlaps between layers")
    
    return two_layer_ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 TWO-LAYER ENSEMBLE READY FOR IMPLEMENTATION!")
        
    except Exception as e:
        print(f"\n❌ Two-layer ensemble building failed: {e}")
        import traceback
        traceback.print_exc()