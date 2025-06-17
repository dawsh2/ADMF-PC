#!/usr/bin/env python3
"""
Build final ensemble from comprehensive results.

Uses the excellent comprehensive analysis results to build the final
regime-adaptive ensemble with proper parameter awareness.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_comprehensive_results():
    """Load the comprehensive results."""
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    file_path = workspace_path / "comprehensive_regime_ensemble.json"
    
    with open(file_path, 'r') as f:
        return json.load(f)


def build_final_optimized_ensemble(results: dict) -> dict:
    """Build final optimized ensemble from comprehensive results."""
    
    print("BUILDING FINAL OPTIMIZED ENSEMBLE FROM COMPREHENSIVE ANALYSIS")
    print("="*70)
    
    regime_champions = results['regime_champions']
    
    # Updated regime frequencies (excluding trending states)
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348, 
        'neutral': 0.185
    }
    
    ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'final_optimized_regime_adaptive',
        'creation_date': pd.Timestamp.now().isoformat(),
        'data_source': 'comprehensive_analysis_all_strategies',
        'exclusions': {
            'trending_regimes': ['bull_trending', 'bear_trending'],
            'reason': 'Insufficient data (<2% combined)',
            'recommendation': 'Adjust classifier grid for better trending detection'
        },
        'regimes': {},
        'expected_performance': {}
    }
    
    total_expected_return = 0
    
    print(f"\n🎯 REGIME SPECIALIST SELECTION:")
    
    # Select top 3 strategies per regime for diversification
    for regime, regime_data in regime_champions.items():
        if regime in ['bull_trending', 'bear_trending']:
            continue  # Skip insufficient data regimes
            
        print(f"\n{regime.upper()} REGIME:")
        
        # Get top 3 performers
        top_strategies = regime_data['top_10'][:3]
        
        # Weight by performance (performance-weighted allocation)
        total_return = sum(s['return'] for s in top_strategies if s['return'] > 0)
        
        selected_strategies = []
        regime_expected_return = 0
        
        for i, strategy in enumerate(top_strategies):
            if strategy['return'] > 0 and total_return > 0:
                weight = strategy['return'] / total_return
            else:
                weight = 1.0 / len(top_strategies)
            
            selected_strategies.append({
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
            
            print(f"  {i+1}. {strategy['strategy_name']}")
            print(f"     Return: {strategy['return']:.2%} | Weight: {weight:.1%} | Trades: {strategy['trades']}")
        
        ensemble['regimes'][regime] = {
            'strategies': selected_strategies,
            'regime_expected_return': regime_expected_return,
            'regime_frequency': regime_frequencies[regime]
        }
        
        # Calculate contribution to total return
        regime_contribution = regime_expected_return * regime_frequencies[regime]
        total_expected_return += regime_contribution
        
        print(f"  📊 Regime expected return: {regime_expected_return:.2%}")
        print(f"  📊 Contribution to total: {regime_contribution:.2%}")
    
    ensemble['expected_performance'] = {
        'total_expected_return': total_expected_return,
        'regime_contributions': {
            regime: data['regime_expected_return'] * data['regime_frequency']
            for regime, data in ensemble['regimes'].items()
        }
    }
    
    print(f"\n🎯 FINAL ENSEMBLE EXPECTED RETURN: {total_expected_return:.2%}")
    
    return ensemble


def create_implementation_config(ensemble: dict) -> dict:
    """Create implementation-ready configuration."""
    
    return {
        'ensemble_name': 'final_optimized_regime_adaptive',
        'classifier': ensemble['classifier'],
        'description': 'Final optimized ensemble from comprehensive analysis of all strategies',
        'regime_strategies': {
            regime: [
                {
                    'strategy_name': s['strategy_name'],
                    'weight': s['weight'],
                    'expected_return': s['expected_return'],
                    'rank': s['rank']
                }
                for s in regime_data['strategies']
            ]
            for regime, regime_data in ensemble['regimes'].items()
        },
        'expected_performance': ensemble['expected_performance'],
        'implementation_notes': {
            'regime_detection': f"Use {ensemble['classifier']} for real-time regime classification",
            'position_sizing': 'Weight positions by strategy weight within current regime',
            'rebalancing': 'Rebalance when regime changes detected',
            'diversification': 'Multiple strategies per regime for robustness'
        }
    }


def analyze_top_performers(results: dict):
    """Analyze the top performers across regimes."""
    
    print(f"\n{'='*80}")
    print("TOP PERFORMERS ANALYSIS")
    print("="*80)
    
    regime_champions = results['regime_champions']
    
    for regime, regime_data in regime_champions.items():
        if regime in ['bull_trending', 'bear_trending']:
            continue
            
        print(f"\n🏆 {regime.upper()} REGIME TOP PERFORMERS:")
        
        champion = regime_data['champion']
        print(f"  Champion: {champion['strategy_name']} ({champion['strategy_type']})")
        print(f"  Return: {champion['return']:.2%}")
        print(f"  Trades: {champion['trades']}")
        print(f"  Win Rate: {champion['win_rate']:.1%}")
        print(f"  Profit Factor: {champion['profit_factor']:.2f}")
        
        # Show strategy type diversity in top 10
        top_10 = regime_data['top_10']
        strategy_types = {}
        for strategy in top_10:
            strategy_type = strategy['strategy_type']
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = []
            strategy_types[strategy_type].append(strategy)
        
        print(f"  Strategy type diversity in top 10:")
        for strategy_type, strategies in strategy_types.items():
            best_return = max(s['return'] for s in strategies)
            print(f"    {strategy_type}: {len(strategies)} strategies (best: {best_return:.2%})")


def main():
    """Main workflow."""
    
    print("FINAL ENSEMBLE BUILDER FROM COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Load comprehensive results
    results = load_comprehensive_results()
    
    # Analyze top performers
    analyze_top_performers(results)
    
    # Build final optimized ensemble
    final_ensemble = build_final_optimized_ensemble(results)
    
    # Create implementation config
    impl_config = create_implementation_config(final_ensemble)
    
    # Save results
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    # Save ensemble analysis
    ensemble_file = workspace_path / "final_optimized_ensemble.json"
    with open(ensemble_file, 'w') as f:
        json.dump({
            'final_ensemble': final_ensemble,
            'source_analysis': 'comprehensive_regime_ensemble.json',
            'methodology': 'Top 3 strategies per regime with performance weighting'
        }, f, indent=2, default=str)
    
    # Save implementation config
    config_file = workspace_path / "final_optimized_ensemble_config.json"
    with open(config_file, 'w') as f:
        json.dump(impl_config, f, indent=2, default=str)
    
    print(f"\n✅ FINAL ENSEMBLE COMPLETE!")
    print(f"  📊 Analysis: {ensemble_file}")
    print(f"  ⚙️  Config: {config_file}")
    print(f"  🎯 Expected return: {final_ensemble['expected_performance']['total_expected_return']:.2%}")
    
    # Summary
    print(f"\n📋 ENSEMBLE SUMMARY:")
    total_strategies = sum(len(regime_data['strategies']) for regime_data in final_ensemble['regimes'].values())
    print(f"  Total strategies: {total_strategies}")
    print(f"  Regimes covered: {list(final_ensemble['regimes'].keys())}")
    print(f"  Diversification: Multiple strategies per regime")
    print(f"  Data coverage: All 1,229 strategies analyzed")
    
    return final_ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 FINAL OPTIMIZED ENSEMBLE READY FOR IMPLEMENTATION!")
        
    except Exception as e:
        print(f"\n❌ Final ensemble building failed: {e}")
        import traceback
        traceback.print_exc()