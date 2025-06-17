#!/usr/bin/env python3
"""
Process comprehensive analysis results to build parameter-aware ensemble.

Takes the output from comprehensive_all_strategies_analysis.py and performs:
1. Parameter neighborhood clustering analysis
2. Profitable region identification  
3. Robust ensemble building from complete dataset
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from typing import Dict, List, Tuple

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def load_comprehensive_results(file_path: Path) -> dict:
    """Load comprehensive analysis results."""
    with open(file_path, 'r') as f:
        return json.load(f)


def parse_strategy_parameters(strategy_name: str) -> Dict[str, float]:
    """
    Parse strategy parameters from filename with improved parsing.
    
    Example: SPY_macd_crossover_grid_5_35_9 -> {'fast': 5, 'slow': 35, 'signal': 9}
    """
    # Remove SPY_ prefix and split
    name_parts = strategy_name.replace('SPY_', '').split('_')
    
    # Extract numeric parameters from the end
    params = {}
    param_values = []
    
    # Work backwards from the end to find numeric parameters
    for part in reversed(name_parts):
        # Handle decimal values like 0.001, 0.05, etc.
        try:
            if '.' in part:
                value = float(part)
            else:
                value = int(part)
            param_values.append(value)
        except ValueError:
            break
    
    param_values.reverse()
    
    # Strategy-specific parameter mapping
    strategy_type = '_'.join(name_parts[:-len(param_values)])
    
    if 'macd' in strategy_type:
        if 'crossover' in strategy_type:
            param_names = ['fast_period', 'slow_period', 'signal_period']
        else:  # momentum
            param_names = ['fast_period', 'slow_period', 'signal_period', 'threshold']
    elif 'rsi' in strategy_type:
        if 'bands' in strategy_type:
            param_names = ['period', 'oversold', 'overbought']
        elif 'threshold' in strategy_type:
            param_names = ['period', 'threshold']
        else:
            param_names = ['k_period', 'rsi_period', 'oversold', 'overbought']
    elif 'stochastic' in strategy_type:
        if 'rsi' in strategy_type:
            param_names = ['k_period', 'rsi_period', 'oversold', 'overbought']
        else:
            param_names = ['k_period', 'd_period']
    elif 'bollinger' in strategy_type:
        param_names = ['period', 'std_dev']
    elif 'williams' in strategy_type:
        param_names = ['period', 'oversold', 'overbought']
    elif any(ma in strategy_type for ma in ['ema', 'sma', 'dema', 'tema']):
        param_names = ['fast_period', 'slow_period']
    elif 'pivot' in strategy_type:
        if 'bounces' in strategy_type:
            param_names = ['period', 'sensitivity', 'threshold']
        else:
            param_names = ['period', 'sensitivity', 'threshold']
    elif 'atr' in strategy_type:
        param_names = ['period', 'lookback', 'multiplier']
    elif 'adx' in strategy_type:
        param_names = ['period', 'threshold', 'strength']
    elif 'cci' in strategy_type:
        if 'bands' in strategy_type:
            param_names = ['period', 'oversold', 'overbought']
        else:
            param_names = ['period', 'threshold']
    elif 'mfi' in strategy_type:
        param_names = ['period', 'oversold', 'overbought']
    elif 'ultimate_oscillator' in strategy_type:
        param_names = ['short_period', 'medium_period', 'long_period', 'oversold', 'overbought']
    elif 'ichimoku' in strategy_type:
        param_names = ['conversion_period', 'base_period']
    elif 'elder_ray' in strategy_type:
        param_names = ['period', 'bull_threshold', 'bear_threshold']
    elif 'roc' in strategy_type:
        param_names = ['period', 'threshold']
    elif 'vortex' in strategy_type:
        param_names = ['period']
    elif 'parabolic_sar' in strategy_type:
        param_names = ['af_start', 'af_increment', 'af_max']
    elif 'donchian' in strategy_type:
        param_names = ['period']
    elif 'keltner' in strategy_type:
        param_names = ['period', 'multiplier']
    elif 'trendline' in strategy_type:
        param_names = ['lookback', 'min_touches', 'threshold', 'strength']
    elif 'chaikin' in strategy_type:
        param_names = ['period', 'threshold']
    elif 'aroon' in strategy_type:
        param_names = ['period', 'threshold'] if len(param_values) > 1 else ['period']
    elif 'momentum_breakout' in strategy_type:
        param_names = ['period', 'threshold']
    elif 'supertrend' in strategy_type:
        param_names = ['period', 'multiplier']
    elif 'fibonacci' in strategy_type:
        param_names = ['lookback']
    elif 'vwap' in strategy_type:
        param_names = ['period', 'threshold']
    elif 'support_resistance' in strategy_type:
        param_names = ['period', 'threshold']
    elif 'linear_regression' in strategy_type:
        param_names = ['period']
    elif 'price_action' in strategy_type:
        param_names = ['lookback']
    else:
        # Generic parameter names
        param_names = [f'param_{i+1}' for i in range(len(param_values))]
    
    # Map values to parameter names
    for i, value in enumerate(param_values):
        if i < len(param_names):
            params[param_names[i]] = value
    
    # Add strategy type info
    params['strategy_type'] = strategy_type
    
    return params


def analyze_comprehensive_parameter_neighborhoods(comprehensive_results: dict) -> Dict[str, dict]:
    """
    Analyze parameter neighborhoods from comprehensive results.
    
    This provides the COMPLETE picture across all 1,229 strategies.
    """
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE PARAMETER NEIGHBORHOOD ANALYSIS")
    print(f"Analyzing {len(comprehensive_results.get('strategies', {}))} strategies")
    print("="*80)
    
    # Group strategies by type with full parameter analysis
    strategy_groups = defaultdict(list)
    
    for strategy_name, strategy_data in comprehensive_results.get('strategies', {}).items():
        # Parse parameters
        params = parse_strategy_parameters(strategy_name)
        strategy_type = params.get('strategy_type', 'unknown')
        
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
            'avg_trade_return': overall_perf.get('avg_trade_return', 0),
            'best_trade': overall_perf.get('best_trade', 0),
            'worst_trade': overall_perf.get('worst_trade', 0),
            'regime_performance': strategy_data.get('regime_performance', {})
        }
        
        strategy_groups[strategy_type].append(strategy_info)
    
    print(f"Found {len(strategy_groups)} strategy types with sufficient data")
    
    # Analyze each strategy type for comprehensive parameter clustering
    comprehensive_neighborhoods = {}
    
    for strategy_type, strategies in strategy_groups.items():
        if len(strategies) < 3:  # Need minimum strategies for analysis
            continue
            
        print(f"\n📊 {strategy_type.upper()}:")
        print(f"   Analyzing {len(strategies)} strategies...")
        
        # Convert to DataFrame for analysis
        df_data = []
        for strategy in strategies:
            row = {
                'name': strategy['name'], 
                'return': strategy['return'],
                'trades': strategy['trades'],
                'win_rate': strategy['win_rate'],
                'profit_factor': strategy['profit_factor']
            }
            # Add parameters (excluding strategy_type)
            for key, value in strategy['parameters'].items():
                if key != 'strategy_type':
                    row[key] = value
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Comprehensive profitability analysis
        profitable = df[df['return'] > 0].copy()
        highly_profitable = df[df['return'] > 0.05].copy()  # >5% return
        
        print(f"   📈 Performance distribution:")
        print(f"     Total strategies: {len(strategies)}")
        print(f"     Profitable (>0%): {len(profitable)} ({len(profitable)/len(strategies)*100:.1f}%)")
        print(f"     Highly profitable (>5%): {len(highly_profitable)} ({len(highly_profitable)/len(strategies)*100:.1f}%)")
        
        if len(profitable) == 0:
            print(f"   ❌ No profitable strategies - skipping")
            continue
        
        # Parameter analysis for profitable strategies
        param_columns = [col for col in df.columns if col not in ['name', 'return', 'trades', 'win_rate', 'profit_factor']]
        param_analysis = {}
        
        for param in param_columns:
            if param in profitable.columns:
                profitable_values = profitable[param].dropna()
                all_values = df[param].dropna()
                
                if len(profitable_values) > 0 and len(all_values) > 1:
                    param_stats = {
                        'profitable_count': len(profitable_values),
                        'total_count': len(all_values),
                        'profitable_min': profitable_values.min(),
                        'profitable_max': profitable_values.max(),
                        'profitable_mean': profitable_values.mean(),
                        'profitable_median': profitable_values.median(),
                        'profitable_std': profitable_values.std(),
                        'all_min': all_values.min(),
                        'all_max': all_values.max(),
                        'all_mean': all_values.mean(),
                        'concentration_ratio': len(profitable_values) / len(all_values)
                    }
                    
                    # Calculate profitable range as percentage of total range
                    total_range = all_values.max() - all_values.min()
                    profitable_range = profitable_values.max() - profitable_values.min()
                    
                    if total_range > 0:
                        param_stats['profitable_range_ratio'] = profitable_range / total_range
                    else:
                        param_stats['profitable_range_ratio'] = 1.0
                    
                    param_analysis[param] = param_stats
                    
                    print(f"     {param}: {param_stats['profitable_count']}/{param_stats['total_count']} profitable "
                          f"(range: {param_stats['profitable_min']:.3f}-{param_stats['profitable_max']:.3f})")
        
        # Find strategies in profitable neighborhoods (comprehensive version)
        neighborhood_strategies = []
        
        for strategy in strategies:
            if strategy['return'] <= 0:  # Only consider profitable strategies for neighborhoods
                continue
                
            neighborhood_score = 0
            total_params = 0
            
            for param, analysis in param_analysis.items():
                if param in strategy['parameters']:
                    value = strategy['parameters'][param]
                    
                    # Check if within profitable range (with some tolerance)
                    profitable_min = analysis['profitable_min']
                    profitable_max = analysis['profitable_max']
                    profitable_mean = analysis['profitable_mean']
                    profitable_std = analysis['profitable_std']
                    
                    total_params += 1
                    
                    # Multiple criteria for being in profitable neighborhood
                    in_range = profitable_min <= value <= profitable_max
                    
                    if profitable_std > 0:
                        # Within 2 standard deviations of profitable mean
                        z_score = abs(value - profitable_mean) / profitable_std
                        in_std_range = z_score <= 2.0
                    else:
                        in_std_range = abs(value - profitable_mean) < 0.001
                    
                    if in_range and in_std_range:
                        neighborhood_score += 1
            
            # Strategy qualifies if most parameters are in profitable neighborhoods
            if total_params > 0:
                neighborhood_ratio = neighborhood_score / total_params
                if neighborhood_ratio >= 0.7:  # 70% of parameters in neighborhoods
                    strategy['neighborhood_score'] = neighborhood_ratio
                    strategy['neighborhood_param_count'] = neighborhood_score
                    neighborhood_strategies.append(strategy)
        
        # Sort by return and take top performers
        neighborhood_strategies.sort(key=lambda x: x['return'], reverse=True)
        
        comprehensive_neighborhoods[strategy_type] = {
            'parameter_analysis': param_analysis,
            'all_strategies': strategies,
            'profitable_strategies': [s for s in strategies if s['return'] > 0],
            'neighborhood_strategies': neighborhood_strategies,
            'performance_stats': {
                'total_count': len(strategies),
                'profitable_count': len(profitable),
                'highly_profitable_count': len(highly_profitable),
                'neighborhood_count': len(neighborhood_strategies),
                'best_return': max(s['return'] for s in strategies),
                'worst_return': min(s['return'] for s in strategies),
                'profitable_rate': len(profitable) / len(strategies)
            }
        }
        
        print(f"   🎯 Neighborhood analysis:")
        print(f"     Strategies in profitable neighborhoods: {len(neighborhood_strategies)}")
        if neighborhood_strategies:
            top_3 = neighborhood_strategies[:3]
            for i, strategy in enumerate(top_3):
                print(f"     {i+1}. {strategy['name']}: {strategy['return']:.2%} "
                      f"(score: {strategy['neighborhood_score']:.2f})")
    
    return comprehensive_neighborhoods


def build_final_robust_ensemble(
    comprehensive_neighborhoods: dict,
    main_regimes: list = ['bull_ranging', 'bear_ranging', 'neutral']
) -> dict:
    """
    Build the final robust ensemble from comprehensive parameter analysis.
    """
    
    print(f"\n{'='*80}")
    print("BUILDING FINAL ROBUST ENSEMBLE FROM COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348,
        'neutral': 0.185
    }
    
    ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'comprehensive_parameter_neighborhood_robust',
        'creation_date': pd.Timestamp.now().isoformat(),
        'classifier_notes': {
            'regimes_used': main_regimes,
            'regimes_excluded': ['bull_trending', 'bear_trending'],
            'exclusion_reason': 'Insufficient data (<2% combined, <1k bars each)',
            'configuration_recommendation': 'Consider adjusting classifier grid parameters to make trending states less restrictive'
        },
        'data_coverage': {
            'total_strategies_analyzed': sum(data['performance_stats']['total_count'] 
                                           for data in comprehensive_neighborhoods.values()),
            'strategy_types_analyzed': len(comprehensive_neighborhoods),
            'methodology': 'Complete dataset analysis with parameter neighborhood validation'
        },
        'regimes': {},
        'expected_performance': {}
    }
    
    total_expected_return = 0
    
    # Select regime specialists from comprehensive neighborhoods
    for regime in main_regimes:
        print(f"\n🎯 {regime.upper()} REGIME SPECIALISTS:")
        
        regime_candidates = []
        
        # Collect candidates from all strategy types
        for strategy_type, neighborhood_data in comprehensive_neighborhoods.items():
            for strategy in neighborhood_data['neighborhood_strategies']:
                regime_perf = strategy['regime_performance'].get(regime)
                
                if regime_perf and regime_perf['trade_count'] >= 30 and regime_perf['net_percentage_return'] > 0:
                    
                    # Calculate risk-adjusted score
                    return_val = regime_perf['net_percentage_return']
                    worst_trade = strategy.get('worst_trade', -0.01)
                    risk_adj_return = return_val / abs(worst_trade) if worst_trade < 0 else return_val
                    
                    candidate = {
                        'strategy_name': strategy['name'],
                        'strategy_type': strategy_type,
                        'regime_return': return_val,
                        'regime_trades': regime_perf['trade_count'],
                        'regime_win_rate': regime_perf['win_rate'],
                        'regime_profit_factor': regime_perf.get('profit_factor', 0),
                        'overall_return': strategy['return'],
                        'neighborhood_score': strategy['neighborhood_score'],
                        'risk_adjusted_return': risk_adj_return,
                        'parameters': strategy['parameters']
                    }
                    regime_candidates.append(candidate)
        
        # Sort by regime return and select top 3
        regime_candidates.sort(key=lambda x: x['regime_return'], reverse=True)
        selected_specialists = regime_candidates[:3]
        
        if selected_specialists:
            # Weight by performance
            total_return = sum(s['regime_return'] for s in selected_specialists)
            for specialist in selected_specialists:
                specialist['weight'] = specialist['regime_return'] / total_return if total_return > 0 else 1/len(selected_specialists)
            
            regime_expected_return = sum(
                s['regime_return'] * s['weight'] for s in selected_specialists
            )
            
            ensemble['regimes'][regime] = {
                'specialists': [
                    {
                        'strategy_name': s['strategy_name'],
                        'strategy_type': s['strategy_type'],
                        'expected_return': s['regime_return'],
                        'weight': s['weight'],
                        'trades': s['regime_trades'],
                        'win_rate': s['regime_win_rate'],
                        'profit_factor': s['regime_profit_factor'],
                        'neighborhood_score': s['neighborhood_score'],
                        'risk_adjusted_return': s['risk_adjusted_return']
                    }
                    for s in selected_specialists
                ],
                'regime_expected_return': regime_expected_return,
                'regime_frequency': regime_frequencies[regime]
            }
            
            regime_contribution = regime_expected_return * regime_frequencies[regime]
            total_expected_return += regime_contribution
            
            for i, specialist in enumerate(selected_specialists):
                print(f"  {i+1}. {specialist['strategy_name']}")
                print(f"     Return: {specialist['regime_return']:.2%} | "
                      f"Weight: {specialist['weight']:.1%} | "
                      f"Trades: {specialist['regime_trades']}")
                print(f"     Neighborhood Score: {specialist['neighborhood_score']:.2f} | "
                      f"Risk-Adj: {specialist['risk_adjusted_return']:.2f}")
        else:
            print(f"  ❌ No qualified specialists found")
    
    ensemble['expected_performance'] = {
        'total_expected_return': total_expected_return,
        'regime_contributions': {
            regime: data['regime_expected_return'] * data['regime_frequency']
            for regime, data in ensemble['regimes'].items()
        },
        'robustness_metrics': {
            'data_completeness': 'Full 1,229 strategy analysis',
            'parameter_validation': 'Neighborhood clustering validation',
            'outlier_filtering': 'Excluded isolated parameter outliers'
        }
    }
    
    print(f"\n🎯 FINAL ROBUST ENSEMBLE EXPECTED RETURN: {total_expected_return:.2%}")
    
    return ensemble


def main():
    """Main comprehensive results processing workflow."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    input_file = workspace_path / "comprehensive_all_strategies_analysis.json"
    
    print("COMPREHENSIVE RESULTS PROCESSOR")
    print("="*50)
    
    # Check if comprehensive results exist
    if not input_file.exists():
        print(f"❌ Comprehensive results not found: {input_file}")
        print("Please run comprehensive_all_strategies_analysis.py first")
        return
    
    print(f"📊 Loading comprehensive results from: {input_file}")
    
    # Load comprehensive results
    try:
        comprehensive_results = load_comprehensive_results(input_file)
        print(f"✅ Loaded results with {len(comprehensive_results.get('strategies', {}))} strategies")
    except Exception as e:
        print(f"❌ Failed to load comprehensive results: {e}")
        return
    
    # Analyze parameter neighborhoods from complete dataset
    comprehensive_neighborhoods = analyze_comprehensive_parameter_neighborhoods(comprehensive_results)
    
    # Build final robust ensemble
    final_ensemble = build_final_robust_ensemble(comprehensive_neighborhoods)
    
    # Save processed results
    output_file = workspace_path / "final_robust_ensemble_analysis.json"
    
    processed_results = {
        'final_robust_ensemble': final_ensemble,
        'comprehensive_parameter_neighborhoods': comprehensive_neighborhoods,
        'source_analysis': str(input_file),
        'processing_metadata': {
            'total_strategies_processed': len(comprehensive_results.get('strategies', {})),
            'strategy_types_with_neighborhoods': len(comprehensive_neighborhoods),
            'methodology': 'Comprehensive parameter neighborhood analysis from complete dataset'
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(processed_results, f, indent=2, default=str)
    
    # Create final implementation config
    impl_config = {
        'ensemble_name': 'final_robust_regime_adaptive',
        'classifier': final_ensemble['classifier'],
        'data_coverage': final_ensemble['data_coverage'],
        'regime_strategies': {
            regime: [
                {
                    'strategy_name': s['strategy_name'],
                    'weight': s['weight'],
                    'expected_return': s['expected_return'],
                    'neighborhood_validation': s['neighborhood_score']
                }
                for s in regime_data['specialists']
            ]
            for regime, regime_data in final_ensemble['regimes'].items()
        },
        'expected_performance': final_ensemble['expected_performance']
    }
    
    impl_config_file = workspace_path / "final_robust_ensemble_config.json"
    with open(impl_config_file, 'w') as f:
        json.dump(impl_config, f, indent=2, default=str)
    
    print(f"\n✅ COMPREHENSIVE PROCESSING COMPLETE!")
    print(f"  📊 Analysis: {output_file}")
    print(f"  ⚙️  Config: {impl_config_file}")
    print(f"  🎯 Expected return: {final_ensemble['expected_performance']['total_expected_return']:.2%}")
    print(f"  🔬 Methodology: Complete dataset parameter neighborhood analysis")
    
    return final_ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 FINAL ROBUST ENSEMBLE READY!")
        
    except Exception as e:
        print(f"\n❌ Comprehensive results processing failed: {e}")
        import traceback
        traceback.print_exc()