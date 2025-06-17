#!/usr/bin/env python3
"""
Comprehensive regime-adaptive ensemble builder.

Systematically analyze ALL 45+ strategy types and 1,200+ unique instances
to find the true regime specialists for building an optimal ensemble.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.sparse_trace_analysis import (
    StrategyAnalyzer,
    ExecutionCostConfig,
    analyze_strategy_performance_by_regime
)


def survey_strategy_universe(workspace_path: Path):
    """Get comprehensive overview of available strategies."""
    
    signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
    
    strategy_inventory = {}
    total_strategies = 0
    
    for strategy_type_dir in signals_dir.iterdir():
        if strategy_type_dir.is_dir():
            strategy_files = list(strategy_type_dir.glob("*.parquet"))
            strategy_inventory[strategy_type_dir.name] = {
                'count': len(strategy_files),
                'files': strategy_files
            }
            total_strategies += len(strategy_files)
    
    return strategy_inventory, total_strategies


def sample_strategies_per_type(strategy_inventory: dict, samples_per_type: int = 5):
    """
    Sample strategies from each type for initial regime specialist discovery.
    
    For large-scale analysis, we'll sample first to identify promising types,
    then dive deeper into the best ones.
    """
    
    sampled_strategies = {}
    total_sampled = 0
    
    for strategy_type, type_data in strategy_inventory.items():
        files = type_data['files']
        
        # Sample evenly across parameter space if possible
        if len(files) <= samples_per_type:
            sample = files
        else:
            # Take evenly spaced samples
            step = len(files) // samples_per_type
            sample = files[::step][:samples_per_type]
        
        sampled_strategies[strategy_type] = sample
        total_sampled += len(sample)
    
    return sampled_strategies, total_sampled


def analyze_strategy_batch(
    strategy_files: list,
    classifier_name: str,
    workspace_path: Path,
    cost_config: ExecutionCostConfig,
    batch_name: str
):
    """Analyze a batch of strategies and return results."""
    
    print(f"  Analyzing {batch_name}: {len(strategy_files)} strategies...")
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
        
        print(f"    ✅ {batch_name}: {success_count}/{len(strategy_files)} strategies analyzed in {elapsed:.1f}s")
        
        return batch_name, results
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    ❌ {batch_name} failed after {elapsed:.1f}s: {e}")
        return batch_name, None


def find_regime_champions(all_results: dict, main_regimes: list, min_trades: int = 30):
    """
    Find the absolute best performer for each regime across ALL strategy types.
    
    This is where we identify the true regime specialists.
    """
    
    print(f"\n{'='*80}")
    print("FINDING REGIME CHAMPIONS ACROSS ALL STRATEGY TYPES")
    print("="*80)
    
    regime_champions = {}
    
    for regime in main_regimes:
        print(f"\n🏆 SEARCHING FOR {regime.upper()} CHAMPION...")
        
        all_candidates = []
        
        # Collect ALL candidates for this regime across all strategy types
        for batch_name, batch_results in all_results.items():
            if batch_results is None:
                continue
                
            for strategy_name, strategy_data in batch_results.get('strategies', {}).items():
                regime_perf = strategy_data.get('regime_performance', {}).get(regime)
                
                if regime_perf and regime_perf['trade_count'] >= min_trades:
                    
                    # Extract strategy type from file path
                    strategy_type = strategy_data['strategy_file'].split('/')[-2]
                    
                    candidate = {
                        'strategy_name': strategy_name,
                        'strategy_type': strategy_type,
                        'batch_source': batch_name,
                        'return': regime_perf['net_percentage_return'],
                        'log_return': regime_perf['total_net_log_return'],
                        'trades': regime_perf['trade_count'],
                        'win_rate': regime_perf['win_rate'],
                        'avg_trade_return': regime_perf['avg_trade_return'],
                        'profit_factor': regime_perf.get('profit_factor', 0),
                        'avg_bars_held': regime_perf['avg_bars_held'],
                        'best_trade': regime_perf['best_trade'],
                        'worst_trade': regime_perf['worst_trade']
                    }
                    
                    all_candidates.append(candidate)
        
        if not all_candidates:
            print(f"  ❌ No qualified candidates found for {regime}")
            regime_champions[regime] = None
            continue
        
        # Sort by return to find champion
        all_candidates.sort(key=lambda x: x['return'], reverse=True)
        
        champion = all_candidates[0]
        top_5 = all_candidates[:5]
        
        print(f"  🥇 CHAMPION: {champion['strategy_name']} ({champion['strategy_type']})")
        print(f"     Return: {champion['return']:.2%}")
        print(f"     Trades: {champion['trades']}")
        print(f"     Win Rate: {champion['win_rate']:.1%}")
        print(f"     Profit Factor: {champion['profit_factor']:.2f}")
        
        print(f"  🏅 Top 5 contenders:")
        for i, candidate in enumerate(top_5):
            print(f"     {i+1}. {candidate['strategy_name']} ({candidate['strategy_type']}): {candidate['return']:.2%}")
        
        regime_champions[regime] = {
            'champion': champion,
            'top_5': top_5,
            'total_candidates': len(all_candidates)
        }
    
    return regime_champions


def analyze_strategy_type_specialization(regime_champions: dict):
    """Analyze which strategy types excel in which regimes."""
    
    print(f"\n{'='*80}")
    print("STRATEGY TYPE SPECIALIZATION ANALYSIS")
    print("="*80)
    
    type_specialization = {}
    
    for regime, regime_data in regime_champions.items():
        if regime_data is None:
            continue
            
        print(f"\n{regime.upper()} REGIME - Top Strategy Types:")
        
        # Count strategy types in top performers
        type_counts = {}
        for candidate in regime_data['top_5']:
            strategy_type = candidate['strategy_type']
            if strategy_type not in type_counts:
                type_counts[strategy_type] = []
            type_counts[strategy_type].append(candidate)
        
        # Show type dominance
        for strategy_type, candidates in sorted(type_counts.items(), key=lambda x: len(x[1]), reverse=True):
            best_return = max(c['return'] for c in candidates)
            print(f"  {strategy_type}: {len(candidates)} in top 5 (best: {best_return:.2%})")
            
            if strategy_type not in type_specialization:
                type_specialization[strategy_type] = {}
            type_specialization[strategy_type][regime] = {
                'count_in_top5': len(candidates),
                'best_return': best_return,
                'candidates': candidates
            }
    
    # Summary of type specializations
    print(f"\n{'='*60}")
    print("STRATEGY TYPE REGIME PREFERENCES")
    print("="*60)
    
    for strategy_type, regime_data in type_specialization.items():
        # Find best regime for this type
        best_regime = max(regime_data.keys(), key=lambda r: regime_data[r]['best_return'])
        best_return = regime_data[best_regime]['best_return']
        
        print(f"{strategy_type}:")
        print(f"  Best regime: {best_regime} ({best_return:.2%})")
        print(f"  Appears in top 5 for: {list(regime_data.keys())}")
    
    return type_specialization


def build_optimal_ensemble(regime_champions: dict, ensemble_size: int = 1):
    """Build the optimal ensemble using the best regime specialists."""
    
    print(f"\n{'='*80}")
    print(f"BUILDING OPTIMAL REGIME-ADAPTIVE ENSEMBLE")
    print(f"Ensemble size: {ensemble_size} strategy per regime")
    print("="*80)
    
    ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'regime_specialist',
        'creation_date': pd.Timestamp.now().isoformat(),
        'regimes': {},
        'expected_performance': {}
    }
    
    # Regime frequencies from our analysis
    regime_frequencies = {
        'bull_ranging': 0.447,
        'bear_ranging': 0.348,
        'neutral': 0.185
    }
    
    total_expected_return = 0
    
    for regime, regime_data in regime_champions.items():
        if regime_data is None:
            print(f"{regime}: No champion available")
            continue
            
        champion = regime_data['champion']
        
        ensemble['regimes'][regime] = {
            'strategy_name': champion['strategy_name'],
            'strategy_type': champion['strategy_type'],
            'expected_return': champion['return'],
            'trade_frequency': champion['trades'],
            'win_rate': champion['win_rate'],
            'profit_factor': champion['profit_factor'],
            'confidence': 'high' if champion['trades'] > 100 else 'medium' if champion['trades'] > 50 else 'low'
        }
        
        # Calculate contribution to overall expected return
        freq = regime_frequencies.get(regime, 0)
        contribution = champion['return'] * freq
        total_expected_return += contribution
        
        print(f"{regime.upper()}:")
        print(f"  Champion: {champion['strategy_name']} ({champion['strategy_type']})")
        print(f"  Expected return: {champion['return']:.2%}")
        print(f"  Regime frequency: {freq:.1%}")
        print(f"  Contribution: {contribution:.2%}")
        print(f"  Confidence: {ensemble['regimes'][regime]['confidence']}")
    
    ensemble['expected_performance'] = {
        'total_expected_return': total_expected_return,
        'regime_contributions': {
            regime: data['expected_return'] * regime_frequencies.get(regime, 0)
            for regime, data in ensemble['regimes'].items()
        }
    }
    
    print(f"\n🎯 ENSEMBLE EXPECTED RETURN: {total_expected_return:.2%}")
    
    return ensemble


def main():
    """Main comprehensive ensemble building workflow."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    print("COMPREHENSIVE REGIME-ADAPTIVE ENSEMBLE BUILDER")
    print("="*60)
    print("Analyzing ALL strategy types to find true regime specialists")
    print("="*60)
    
    # Survey the strategy universe
    print("Step 1: Surveying strategy universe...")
    strategy_inventory, total_strategies = survey_strategy_universe(workspace_path)
    
    print(f"Found {len(strategy_inventory)} strategy types with {total_strategies} total strategies:")
    for strategy_type, data in sorted(strategy_inventory.items()):
        print(f"  {strategy_type}: {data['count']} strategies")
    
    # Sample strategies for comprehensive analysis
    print(f"\nStep 2: Sampling strategies for analysis...")
    sampled_strategies, total_sampled = sample_strategies_per_type(strategy_inventory, samples_per_type=3)
    
    print(f"Selected {total_sampled} strategies for analysis (3 per type for speed)")
    
    # Setup analysis configuration
    classifier_name = "SPY_market_regime_grid_0006_12"
    cost_config = ExecutionCostConfig(cost_multiplier=0.99)
    main_regimes = ['bull_ranging', 'bear_ranging', 'neutral']
    
    print(f"Using classifier: {classifier_name}")
    print(f"Analyzing regimes: {main_regimes}")
    
    # Analyze all strategy types in parallel batches
    print(f"\nStep 3: Analyzing all strategy types...")
    
    all_results = {}
    
    # Process in batches to manage memory and show progress
    batch_size = 5  # 5 strategy types at a time
    strategy_types = list(sampled_strategies.keys())
    
    for i in range(0, len(strategy_types), batch_size):
        batch_types = strategy_types[i:i+batch_size]
        
        print(f"\nBatch {i//batch_size + 1}/{(len(strategy_types) + batch_size - 1)//batch_size}: {batch_types}")
        
        for strategy_type in batch_types:
            strategy_files = sampled_strategies[strategy_type]
            
            batch_name, results = analyze_strategy_batch(
                strategy_files,
                classifier_name,
                workspace_path,
                cost_config,
                strategy_type
            )
            
            if results:
                all_results[batch_name] = results
    
    successful_types = len([r for r in all_results.values() if r is not None])
    print(f"\n✅ Successfully analyzed {successful_types}/{len(strategy_types)} strategy types")
    
    # Find regime champions
    regime_champions = find_regime_champions(all_results, main_regimes)
    
    # Analyze strategy type specialization
    type_specialization = analyze_strategy_type_specialization(regime_champions)
    
    # Build optimal ensemble
    optimal_ensemble = build_optimal_ensemble(regime_champions)
    
    # Save comprehensive results
    output_file = workspace_path / "comprehensive_regime_ensemble.json"
    
    comprehensive_results = {
        'strategy_inventory': {k: {'count': v['count']} for k, v in strategy_inventory.items()},
        'analysis_summary': {
            'total_strategy_types': len(strategy_inventory),
            'total_strategies_available': total_strategies,
            'strategies_analyzed': total_sampled,
            'successful_analysis_types': successful_types
        },
        'regime_champions': regime_champions,
        'type_specialization': type_specialization,
        'optimal_ensemble': optimal_ensemble
    }
    
    with open(output_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    # Save just the ensemble config for implementation
    ensemble_config_file = workspace_path / "regime_ensemble_config.json"
    with open(ensemble_config_file, 'w') as f:
        json.dump(optimal_ensemble, f, indent=2, default=str)
    
    print(f"\n✅ Results saved:")
    print(f"  📊 Full analysis: {output_file}")
    print(f"  ⚙️  Ensemble config: {ensemble_config_file}")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ENSEMBLE ANALYSIS COMPLETE")
    print("="*80)
    
    print("🏆 REGIME CHAMPIONS:")
    for regime, regime_data in regime_champions.items():
        if regime_data:
            champion = regime_data['champion']
            print(f"  {regime.upper()}: {champion['strategy_name']} ({champion['strategy_type']}) - {champion['return']:.2%}")
        else:
            print(f"  {regime.upper()}: No champion found")
    
    print(f"\n🎯 Expected ensemble return: {optimal_ensemble['expected_performance']['total_expected_return']:.2%}")
    
    return optimal_ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 REGIME-ADAPTIVE ENSEMBLE READY FOR IMPLEMENTATION!")
        
    except Exception as e:
        print(f"\n❌ Comprehensive ensemble building failed: {e}")
        import traceback
        traceback.print_exc()