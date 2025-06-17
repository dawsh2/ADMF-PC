#!/usr/bin/env python3
"""
Smart Parameter-Aware Ensemble Builder

Focuses on parameter clustering and profitability neighborhoods rather than
exhaustive analysis. Identifies robust strategies that live in profitable
parameter regions, not just random outliers.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import time
import re
from collections import defaultdict
from typing import Dict, List, Tuple

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.sparse_trace_analysis import (
    StrategyAnalyzer,
    ExecutionCostConfig
)


def parse_strategy_parameters(strategy_name: str) -> Dict[str, float]:
    """
    Parse strategy parameters from filename.
    
    Example: SPY_macd_crossover_grid_5_35_9 -> {'fast': 5, 'slow': 35, 'signal': 9}
    """
    parts = strategy_name.split('_')
    
    # Extract numeric parameters from the end
    params = {}
    param_values = []
    
    for part in reversed(parts):
        try:
            # Try to parse as float (handles decimals like 0.001)
            value = float(part)
            param_values.append(value)
        except ValueError:
            break
    
    param_values.reverse()
    
    # Map to generic parameter names based on strategy type
    if 'macd' in strategy_name:
        param_names = ['fast_period', 'slow_period', 'signal_period']
    elif 'rsi' in strategy_name:
        param_names = ['period', 'oversold', 'overbought'] 
    elif 'bollinger' in strategy_name:
        param_names = ['period', 'std_dev']
    elif 'ema' in strategy_name or 'sma' in strategy_name:
        param_names = ['fast_period', 'slow_period']
    elif 'stochastic' in strategy_name:
        param_names = ['k_period', 'd_period']
    elif 'williams' in strategy_name:
        param_names = ['period', 'oversold', 'overbought']
    elif 'pivot' in strategy_name:
        param_names = ['period', 'sensitivity', 'threshold']
    else:
        # Generic parameter names
        param_names = [f'param_{i+1}' for i in range(len(param_values))]
    
    # Map values to parameter names
    for i, value in enumerate(param_values):
        if i < len(param_names):
            params[param_names[i]] = value
    
    return params


def analyze_parameter_neighborhoods(strategy_results: dict) -> Dict[str, dict]:
    """
    Analyze parameter neighborhoods to find profitable clusters.
    
    Returns strategies that live in profitable parameter regions,
    not just random outliers.
    """
    
    print(f"\n{'='*80}")
    print("PARAMETER NEIGHBORHOOD ANALYSIS")
    print("="*80)
    
    # Group strategies by type
    strategy_groups = defaultdict(list)
    
    for strategy_name, strategy_data in strategy_results.items():
        strategy_type = strategy_data['strategy_file'].split('/')[-2]
        
        # Parse parameters
        params = parse_strategy_parameters(strategy_name)
        
        # Get overall performance
        overall_perf = strategy_data.get('overall_performance', {})
        if not overall_perf or overall_perf.get('trade_count', 0) < 30:
            continue
        
        strategy_info = {
            'name': strategy_name,
            'parameters': params,
            'return': overall_perf.get('net_percentage_return', 0),
            'trades': overall_perf.get('trade_count', 0),
            'win_rate': overall_perf.get('win_rate', 0),
            'profit_factor': overall_perf.get('profit_factor', 0),
            'regime_performance': strategy_data.get('regime_performance', {})
        }
        
        strategy_groups[strategy_type].append(strategy_info)
    
    # Analyze each strategy type for parameter clustering
    profitable_neighborhoods = {}
    
    for strategy_type, strategies in strategy_groups.items():
        if len(strategies) < 5:  # Need enough strategies to analyze neighborhoods
            continue
            
        print(f"\n📊 {strategy_type.upper()}:")
        print(f"   Analyzing {len(strategies)} strategies for parameter clustering...")
        
        # Convert to DataFrame for easier analysis
        df_data = []
        for strategy in strategies:
            row = {'name': strategy['name'], 'return': strategy['return']}
            row.update(strategy['parameters'])
            df_data.append(row)
        
        if not df_data:
            continue
            
        df = pd.DataFrame(df_data)
        
        # Find top performers (>0% return)
        profitable = df[df['return'] > 0].copy()
        
        if len(profitable) == 0:
            print(f"   ❌ No profitable strategies found")
            continue
        
        print(f"   ✅ {len(profitable)}/{len(strategies)} strategies are profitable")
        
        # Analyze parameter ranges of profitable strategies
        param_analysis = {}
        param_columns = [col for col in df.columns if col not in ['name', 'return']]
        
        for param in param_columns:
            if param in profitable.columns:
                profitable_values = profitable[param].dropna()
                all_values = df[param].dropna()
                
                if len(profitable_values) > 0:
                    param_analysis[param] = {
                        'profitable_min': profitable_values.min(),
                        'profitable_max': profitable_values.max(),
                        'profitable_mean': profitable_values.mean(),
                        'profitable_std': profitable_values.std(),
                        'all_min': all_values.min(),
                        'all_max': all_values.max(),
                        'concentration_score': len(profitable_values) / len(all_values)
                    }
        
        # Find strategies in profitable neighborhoods (not outliers)
        neighborhood_strategies = []
        
        for strategy in strategies:
            # Check if strategy parameters are in profitable neighborhoods
            in_neighborhood = True
            neighborhood_score = 0
            
            for param, analysis in param_analysis.items():
                if param in strategy['parameters']:
                    value = strategy['parameters'][param]
                    
                    # Check if value is within 1 std dev of profitable mean
                    profitable_mean = analysis['profitable_mean']
                    profitable_std = analysis['profitable_std']
                    
                    if profitable_std > 0:
                        z_score = abs(value - profitable_mean) / profitable_std
                        if z_score <= 1.5:  # Within 1.5 std devs
                            neighborhood_score += 1
                    else:
                        # If no std dev, check if equal to mean
                        if abs(value - profitable_mean) < 0.001:
                            neighborhood_score += 1
            
            # Strategy must be in neighborhood for most parameters
            if len(param_analysis) > 0:
                neighborhood_ratio = neighborhood_score / len(param_analysis)
                if neighborhood_ratio >= 0.6:  # 60% of parameters in profitable neighborhoods
                    strategy['neighborhood_score'] = neighborhood_ratio
                    neighborhood_strategies.append(strategy)
        
        # Sort by return and take top performers from neighborhoods
        neighborhood_strategies.sort(key=lambda x: x['return'], reverse=True)
        top_neighborhood = neighborhood_strategies[:5]  # Top 5 from neighborhoods
        
        if top_neighborhood:
            profitable_neighborhoods[strategy_type] = {
                'parameter_analysis': param_analysis,
                'neighborhood_strategies': top_neighborhood,
                'total_strategies': len(strategies),
                'profitable_strategies': len(profitable),
                'neighborhood_strategies_count': len(neighborhood_strategies)
            }
            
            print(f"   📈 Top neighborhood performers:")
            for i, strategy in enumerate(top_neighborhood):
                print(f"     {i+1}. {strategy['name']}: {strategy['return']:.2%} "
                      f"(neighborhood score: {strategy['neighborhood_score']:.2f})")
    
    return profitable_neighborhoods


def select_robust_regime_specialists(
    profitable_neighborhoods: dict,
    classifier_name: str,
    main_regimes: list
) -> Dict[str, list]:
    """
    Select robust regime specialists from profitable parameter neighborhoods.
    """
    
    print(f"\n{'='*80}")
    print("SELECTING ROBUST REGIME SPECIALISTS")
    print("="*80)
    
    regime_specialists = {regime: [] for regime in main_regimes}
    
    for regime in main_regimes:
        print(f"\n🎯 {regime.upper()} REGIME SPECIALISTS:")
        
        regime_candidates = []
        
        # Collect candidates from all strategy types
        for strategy_type, neighborhood_data in profitable_neighborhoods.items():
            for strategy in neighborhood_data['neighborhood_strategies']:
                regime_perf = strategy['regime_performance'].get(regime)
                
                if regime_perf and regime_perf['trade_count'] >= 30:
                    candidate = {
                        'strategy_name': strategy['name'],
                        'strategy_type': strategy_type,
                        'regime_return': regime_perf['net_percentage_return'],
                        'regime_trades': regime_perf['trade_count'],
                        'regime_win_rate': regime_perf['win_rate'],
                        'regime_profit_factor': regime_perf.get('profit_factor', 0),
                        'overall_return': strategy['return'],
                        'neighborhood_score': strategy['neighborhood_score'],
                        'parameters': strategy['parameters']
                    }
                    regime_candidates.append(candidate)
        
        # Sort by regime return and select top performers
        regime_candidates.sort(key=lambda x: x['regime_return'], reverse=True)
        
        # Take top 3 performers with positive returns
        selected_specialists = [c for c in regime_candidates[:10] if c['regime_return'] > 0][:3]
        
        if selected_specialists:
            # Weight by performance (simple approach)
            total_return = sum(s['regime_return'] for s in selected_specialists)
            for specialist in selected_specialists:
                specialist['weight'] = specialist['regime_return'] / total_return if total_return > 0 else 1/len(selected_specialists)
            
            regime_specialists[regime] = selected_specialists
            
            for i, specialist in enumerate(selected_specialists):
                print(f"  {i+1}. {specialist['strategy_name']}")
                print(f"     Return: {specialist['regime_return']:.2%} | "
                      f"Trades: {specialist['regime_trades']} | "
                      f"Weight: {specialist['weight']:.1%}")
                print(f"     Neighborhood score: {specialist['neighborhood_score']:.2f}")
        else:
            print(f"  ❌ No qualified specialists found")
    
    return regime_specialists


def build_parameter_aware_ensemble(
    regime_specialists: dict,
    profitable_neighborhoods: dict
) -> dict:
    """
    Build ensemble based on parameter neighborhood analysis.
    """
    
    print(f"\n{'='*80}")
    print("BUILDING PARAMETER-AWARE ENSEMBLE")
    print("="*80)
    
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348,
        'neutral': 0.185
    }
    
    ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'parameter_aware_regime_adaptive',
        'creation_date': pd.Timestamp.now().isoformat(),
        'methodology': {
            'approach': 'Parameter neighborhood clustering',
            'rationale': 'Select strategies from profitable parameter regions, not random outliers',
            'robustness': 'High - strategies validated across parameter neighborhoods'
        },
        'regimes': {},
        'expected_performance': {}
    }
    
    total_expected_return = 0
    
    for regime, specialists in regime_specialists.items():
        if not specialists:
            continue
        
        regime_expected_return = sum(
            s['regime_return'] * s['weight'] for s in specialists
        )
        
        ensemble['regimes'][regime] = {
            'specialists': [
                {
                    'strategy_name': s['strategy_name'],
                    'strategy_type': s['strategy_type'],
                    'expected_return': s['regime_return'],
                    'weight': s['weight'],
                    'neighborhood_score': s['neighborhood_score'],
                    'parameters': s['parameters']
                }
                for s in specialists
            ],
            'regime_expected_return': regime_expected_return,
            'regime_frequency': regime_frequencies[regime]
        }
        
        regime_contribution = regime_expected_return * regime_frequencies[regime]
        total_expected_return += regime_contribution
        
        print(f"{regime.upper()}:")
        print(f"  Expected return: {regime_expected_return:.2%}")
        print(f"  Contribution: {regime_contribution:.2%}")
    
    ensemble['expected_performance'] = {
        'total_expected_return': total_expected_return,
        'regime_contributions': {
            regime: data['regime_expected_return'] * data['regime_frequency']
            for regime, data in ensemble['regimes'].items()
        },
        'robustness_factors': {
            'parameter_neighborhood_validation': 'Strategies selected from profitable parameter clusters',
            'outlier_filtering': 'Avoided strategies that are isolated parameter outliers',
            'neighborhood_scoring': 'Validated strategies live in profitable neighborhoods'
        }
    }
    
    print(f"\n🎯 PARAMETER-AWARE ENSEMBLE EXPECTED RETURN: {total_expected_return:.2%}")
    
    return ensemble


def main():
    """Main smart parameter-aware ensemble building workflow."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    print("SMART PARAMETER-AWARE ENSEMBLE BUILDER")
    print("="*60)
    print("Focus: Profitable parameter neighborhoods, not random outliers")
    print("="*60)
    
    # Configuration
    classifier_name = "SPY_market_regime_grid_0006_12"
    cost_config = ExecutionCostConfig(cost_multiplier=0.99)
    main_regimes = ['bull_ranging', 'bear_ranging', 'neutral']
    
    # Use smart sampling: 10 strategies per type to get parameter coverage
    print("Step 1: Smart sampling for parameter coverage...")
    
    signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
    sampled_files = []
    
    for strategy_type_dir in signals_dir.iterdir():
        if strategy_type_dir.is_dir():
            strategy_files = list(strategy_type_dir.glob("*.parquet"))
            
            if len(strategy_files) <= 10:
                sampled_files.extend(strategy_files)
            else:
                # Sample evenly across parameter space
                step = len(strategy_files) // 10
                sampled_files.extend(strategy_files[::step][:10])
    
    print(f"Selected {len(sampled_files)} strategies for parameter analysis")
    
    # Analyze selected strategies
    print(f"\nStep 2: Analyzing strategies...")
    analyzer = StrategyAnalyzer(workspace_path)
    
    results = analyzer.analyze_multiple_strategies(
        sampled_files,
        classifier_name,
        cost_config
    )
    
    if not results.get('strategies'):
        print("❌ No strategies analyzed successfully")
        return
    
    print(f"✅ {len(results['strategies'])} strategies analyzed")
    
    # Analyze parameter neighborhoods
    profitable_neighborhoods = analyze_parameter_neighborhoods(results['strategies'])
    
    # Select robust regime specialists
    regime_specialists = select_robust_regime_specialists(
        profitable_neighborhoods, classifier_name, main_regimes
    )
    
    # Build parameter-aware ensemble
    ensemble = build_parameter_aware_ensemble(regime_specialists, profitable_neighborhoods)
    
    # Save results
    output_file = workspace_path / "parameter_aware_ensemble.json"
    
    final_results = {
        'parameter_aware_ensemble': ensemble,
        'profitable_neighborhoods': profitable_neighborhoods,
        'regime_specialists': regime_specialists,
        'methodology': {
            'parameter_clustering': 'Analyzed profitable parameter neighborhoods',
            'robustness_validation': 'Selected strategies from parameter clusters, not outliers',
            'efficiency': f'Smart sampling of {len(sampled_files)} strategies vs {len(list(signals_dir.rglob("*.parquet")))} total'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Create implementation config
    impl_config = {
        'ensemble_name': 'parameter_aware_regime_adaptive',
        'classifier': ensemble['classifier'],
        'methodology': ensemble['methodology'],
        'regime_strategies': {
            regime: [
                {
                    'strategy_name': s['strategy_name'],
                    'weight': s['weight'],
                    'expected_return': s['expected_return'],
                    'parameters': s['parameters'],
                    'robustness_score': s['neighborhood_score']
                }
                for s in regime_data['specialists']
            ]
            for regime, regime_data in ensemble['regimes'].items()
        },
        'expected_performance': ensemble['expected_performance']
    }
    
    impl_config_file = workspace_path / "parameter_aware_ensemble_config.json"
    with open(impl_config_file, 'w') as f:
        json.dump(impl_config, f, indent=2, default=str)
    
    print(f"\n✅ PARAMETER-AWARE ENSEMBLE COMPLETE!")
    print(f"  📊 Analysis: {output_file}")
    print(f"  ⚙️  Config: {impl_config_file}")
    print(f"  🎯 Expected return: {ensemble['expected_performance']['total_expected_return']:.2%}")
    print(f"  🔧 Methodology: Parameter neighborhood clustering for robustness")
    
    return ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 SMART PARAMETER-AWARE ENSEMBLE READY!")
        
    except Exception as e:
        print(f"\n❌ Parameter-aware ensemble building failed: {e}")
        import traceback
        traceback.print_exc()