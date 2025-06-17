#!/usr/bin/env python3
"""
Build optimized regime-adaptive ensemble that balances return and risk.

Creates multiple ensemble configurations:
1. Conservative: More diversification, lower risk
2. Aggressive: Higher concentration in top performers
3. Balanced: Optimal risk-return trade-off
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


def analyze_strategy_performance_distribution(regime_champions: dict):
    """Analyze return distribution to understand concentration vs diversification impact."""
    
    print(f"\n{'='*80}")
    print("STRATEGY PERFORMANCE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    for regime, regime_data in regime_champions.items():
        if not regime_data:
            continue
            
        print(f"\n{regime.upper()} REGIME:")
        
        # Get top 10 performers
        top_performers = regime_data.get('top_10', regime_data.get('top_5', []))
        
        if len(top_performers) >= 5:
            returns = [p['return'] for p in top_performers[:10]]
            
            print(f"  Top performer:     {returns[0]:.2%}")
            print(f"  Top 2 average:     {np.mean(returns[:2]):.2%}")
            print(f"  Top 3 average:     {np.mean(returns[:3]):.2%}")
            print(f"  Top 5 average:     {np.mean(returns[:5]):.2%}")
            if len(returns) >= 10:
                print(f"  Top 10 average:    {np.mean(returns[:10]):.2%}")
            
            # Calculate concentration benefit
            top1_vs_top3 = returns[0] - np.mean(returns[:3])
            print(f"  Concentration benefit (Top 1 vs Top 3): {top1_vs_top3:+.2%}")


def build_optimized_ensemble_variants(regime_champions: dict) -> Dict[str, dict]:
    """
    Build multiple ensemble variants with different risk-return profiles.
    
    Returns:
        Dictionary with conservative, balanced, and aggressive ensemble configs
    """
    
    print(f"\n{'='*80}")
    print("BUILDING OPTIMIZED ENSEMBLE VARIANTS")
    print("="*80)
    
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348,
        'neutral': 0.185
    }
    
    variants = {}
    
    # Configuration for each variant
    ensemble_configs = {
        'conservative': {
            'strategies_per_regime': 4,
            'weighting_method': 'equal',
            'description': 'Maximum diversification, equal weighting'
        },
        'balanced': {
            'strategies_per_regime': 3,
            'weighting_method': 'performance_weighted',
            'description': 'Balanced risk-return, performance weighting'
        },
        'aggressive': {
            'strategies_per_regime': 2,
            'weighting_method': 'top_heavy',
            'description': 'Higher concentration, top-heavy weighting'
        }
    }
    
    for variant_name, config in ensemble_configs.items():
        print(f"\n🎯 {variant_name.upper()} ENSEMBLE:")
        print(f"   {config['description']}")
        
        ensemble = {
            'classifier': 'SPY_market_regime_grid_0006_12',
            'ensemble_method': f'{variant_name}_regime_adaptive',
            'creation_date': pd.Timestamp.now().isoformat(),
            'config': config,
            'regimes': {},
            'expected_performance': {}
        }
        
        total_expected_return = 0
        
        for regime, regime_data in regime_champions.items():
            if not regime_data:
                continue
                
            # Get top performers for this regime
            top_performers = regime_data.get('top_10', regime_data.get('top_5', []))
            selected_strategies = top_performers[:config['strategies_per_regime']]
            
            # Apply weighting method
            if config['weighting_method'] == 'equal':
                # Equal weighting
                for strategy in selected_strategies:
                    strategy['weight'] = 1.0 / len(selected_strategies)
                    
            elif config['weighting_method'] == 'performance_weighted':
                # Weight by relative performance (positive returns only)
                positive_performers = [s for s in selected_strategies if s['return'] > 0]
                if positive_performers:
                    total_return = sum(s['return'] for s in positive_performers)
                    for strategy in selected_strategies:
                        if strategy['return'] > 0:
                            strategy['weight'] = strategy['return'] / total_return
                        else:
                            strategy['weight'] = 0.05  # Small weight for negative performers
                else:
                    # Fall back to equal if all negative
                    for strategy in selected_strategies:
                        strategy['weight'] = 1.0 / len(selected_strategies)
                        
            elif config['weighting_method'] == 'top_heavy':
                # Top-heavy: 70% to best, 30% to second best
                if len(selected_strategies) >= 2:
                    selected_strategies[0]['weight'] = 0.70
                    selected_strategies[1]['weight'] = 0.30
                else:
                    selected_strategies[0]['weight'] = 1.0
            
            # Normalize weights to sum to 1
            total_weight = sum(s['weight'] for s in selected_strategies)
            if total_weight > 0:
                for strategy in selected_strategies:
                    strategy['weight'] = strategy['weight'] / total_weight
            
            # Calculate regime expected return
            regime_expected_return = sum(
                strategy['return'] * strategy['weight'] 
                for strategy in selected_strategies
            )
            
            ensemble['regimes'][regime] = {
                'strategies': [
                    {
                        'strategy_name': s['strategy_name'],
                        'strategy_type': s['strategy_type'],
                        'expected_return': s['return'],
                        'weight': s['weight'],
                        'trades': s['trades'],
                        'win_rate': s['win_rate'],
                        'profit_factor': s['profit_factor']
                    }
                    for s in selected_strategies
                ],
                'regime_expected_return': regime_expected_return,
                'regime_frequency': regime_frequencies[regime]
            }
            
            # Contribution to total return
            regime_contribution = regime_expected_return * regime_frequencies[regime]
            total_expected_return += regime_contribution
            
            print(f"\n  {regime.upper()}:")
            for i, strategy in enumerate(selected_strategies):
                print(f"    {i+1}. {strategy['strategy_name']:<40} "
                      f"Return: {strategy['return']:>8.2%} Weight: {strategy['weight']:>6.1%}")
            print(f"    📊 Regime expected return: {regime_expected_return:.2%}")
            print(f"    📊 Contribution to total:  {regime_contribution:.2%}")
        
        ensemble['expected_performance'] = {
            'total_expected_return': total_expected_return,
            'regime_contributions': {
                regime: data['regime_expected_return'] * data['regime_frequency']
                for regime, data in ensemble['regimes'].items()
            }
        }
        
        print(f"\n  🎯 {variant_name.upper()} EXPECTED RETURN: {total_expected_return:.2%}")
        
        variants[variant_name] = ensemble
    
    return variants


def compare_all_approaches(
    simple_ensemble: dict,
    variants: Dict[str, dict]
) -> dict:
    """Compare all ensemble approaches."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ENSEMBLE COMPARISON")
    print("="*80)
    
    approaches = {
        'Simple (1 per regime)': simple_ensemble['expected_performance']['total_expected_return'],
        **{f"{name.title()} ({variant['config']['strategies_per_regime']} per regime)": 
           variant['expected_performance']['total_expected_return'] 
           for name, variant in variants.items()}
    }
    
    # Sort by expected return
    sorted_approaches = sorted(approaches.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Approach':<35} {'Expected Return':<15} {'Strategy Count':<15} {'Risk Profile':<15}")
    print("-" * 85)
    
    risk_profiles = {
        'Simple (1 per regime)': 'High',
        'Aggressive (2 per regime)': 'Medium-High', 
        'Balanced (3 per regime)': 'Medium',
        'Conservative (4 per regime)': 'Low'
    }
    
    strategy_counts = {
        'Simple (1 per regime)': 3,
        'Aggressive (2 per regime)': 6,
        'Balanced (3 per regime)': 9,
        'Conservative (4 per regime)': 12
    }
    
    for approach, expected_return in sorted_approaches:
        risk_profile = risk_profiles.get(approach, 'Unknown')
        strategy_count = strategy_counts.get(approach, 'Unknown')
        print(f"{approach:<35} {expected_return:<15.2%} {strategy_count:<15} {risk_profile:<15}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    best_return = sorted_approaches[0]
    print(f"  • Highest Return: {best_return[0]} ({best_return[1]:.2%})")
    
    # Find best risk-adjusted option (medium return but lower risk)
    balanced_options = [item for item in sorted_approaches if 'Balanced' in item[0] or 'Aggressive' in item[0]]
    if balanced_options:
        recommended = balanced_options[0]
        print(f"  • Recommended: {recommended[0]} ({recommended[1]:.2%})")
        print(f"    - Good balance of return and diversification")
        print(f"    - Multiple strategies per regime for robustness")
        print(f"    - Performance-based weighting")
    
    return {
        'approaches': approaches,
        'ranked_by_return': sorted_approaches,
        'recommendation': balanced_options[0] if balanced_options else sorted_approaches[0]
    }


def create_implementation_configs(variants: Dict[str, dict], output_dir: Path):
    """Create implementation-ready configurations for each variant."""
    
    print(f"\n{'='*80}")
    print("CREATING IMPLEMENTATION CONFIGURATIONS")
    print("="*80)
    
    for variant_name, ensemble in variants.items():
        
        # Create implementation config
        impl_config = {
            'ensemble_name': f'{variant_name}_regime_adaptive',
            'classifier': ensemble['classifier'],
            'description': ensemble['config']['description'],
            'regime_strategies': {
                regime: [
                    {
                        'strategy_name': s['strategy_name'],
                        'weight': s['weight'],
                        'expected_return': s['expected_return'],
                        'confidence': 'high' if s['trades'] > 1000 else 'medium' if s['trades'] > 500 else 'low'
                    }
                    for s in regime_data['strategies']
                ]
                for regime, regime_data in ensemble['regimes'].items()
            },
            'expected_performance': {
                'total_return': ensemble['expected_performance']['total_expected_return'],
                'regime_contributions': ensemble['expected_performance']['regime_contributions']
            },
            'implementation_notes': {
                'regime_detection': 'Use SPY_market_regime_grid_0006_12 for real-time regime classification',
                'position_sizing': 'Weight positions by strategy weight within current regime',
                'rebalancing': 'Rebalance when regime changes detected',
                'risk_management': f'Diversified across {sum(len(r["strategies"]) for r in ensemble["regimes"].values())} strategies'
            }
        }
        
        # Save implementation config
        config_file = output_dir / f"{variant_name}_ensemble_config.json"
        with open(config_file, 'w') as f:
            json.dump(impl_config, f, indent=2, default=str)
        
        print(f"  ✅ {variant_name.title()} config: {config_file.name}")


def main():
    """Main optimized ensemble building workflow."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    results_file = workspace_path / "comprehensive_regime_ensemble.json"
    
    print("OPTIMIZED REGIME-ADAPTIVE ENSEMBLE BUILDER")
    print("="*60)
    print("Creating multiple ensemble variants with different risk-return profiles")
    print("="*60)
    
    # Load comprehensive results
    results = load_comprehensive_results(results_file)
    regime_champions = results['regime_champions']
    simple_ensemble = results['final_ensemble']
    
    # Analyze performance distribution
    analyze_strategy_performance_distribution(regime_champions)
    
    # Build optimized variants
    variants = build_optimized_ensemble_variants(regime_champions)
    
    # Compare all approaches
    comparison = compare_all_approaches(simple_ensemble, variants)
    
    # Create implementation configs
    create_implementation_configs(variants, workspace_path)
    
    # Save comprehensive analysis
    final_results = {
        'ensemble_variants': variants,
        'performance_comparison': comparison,
        'analysis_metadata': {
            'creation_date': pd.Timestamp.now().isoformat(),
            'base_analysis': str(results_file),
            'optimization_approach': 'multi_variant_risk_return_optimization'
        }
    }
    
    output_file = workspace_path / "optimized_ensemble_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n✅ OPTIMIZED ENSEMBLE ANALYSIS COMPLETE!")
    print(f"  📊 Full analysis: {output_file}")
    print(f"  ⚙️  Implementation configs: {workspace_path}/*_ensemble_config.json")
    
    recommended = comparison['recommendation']
    print(f"\n🎯 RECOMMENDED APPROACH: {recommended[0]}")
    print(f"  Expected return: {recommended[1]:.2%}")
    print(f"  Optimal balance of return and risk diversification")
    
    return variants, comparison


if __name__ == "__main__":
    try:
        variants, comparison = main()
        print(f"\n🚀 OPTIMIZED ENSEMBLE VARIANTS READY!")
        
    except Exception as e:
        print(f"\n❌ Optimized ensemble building failed: {e}")
        import traceback
        traceback.print_exc()