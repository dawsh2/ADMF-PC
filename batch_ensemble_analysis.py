#!/usr/bin/env python3
"""
Batched regime-adaptive ensemble analysis for large strategy universes.

This script processes strategy types in small batches, saving intermediate results
to enable incremental analysis of the full 1,200+ strategy universe.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.sparse_trace_analysis import (
    StrategyAnalyzer,
    ExecutionCostConfig
)


def save_batch_results(batch_results: dict, batch_id: int, output_dir: Path):
    """Save batch results with timestamp."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file = output_dir / f"batch_{batch_id:02d}_{timestamp}.json"
    
    with open(batch_file, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    return batch_file


def load_all_batch_results(output_dir: Path) -> dict:
    """Load and combine all batch results."""
    
    batch_files = list(output_dir.glob("batch_*.json"))
    batch_files.sort()  # Process in order
    
    combined_results = {}
    
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            combined_results.update(batch_data)
    
    return combined_results


def analyze_strategy_type_batch(
    strategy_types: list,
    workspace_path: Path,
    classifier_name: str,
    cost_config: ExecutionCostConfig,
    samples_per_type: int = 2,
    batch_id: int = None
):
    """Analyze a batch of strategy types."""
    
    print(f"Batch {batch_id}: Processing {len(strategy_types)} strategy types...")
    
    signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
    analyzer = StrategyAnalyzer(workspace_path)
    
    batch_results = {}
    
    for strategy_type in strategy_types:
        print(f"  Analyzing {strategy_type}...")
        
        strategy_type_dir = signals_dir / strategy_type
        if not strategy_type_dir.exists():
            print(f"    ❌ Directory not found: {strategy_type_dir}")
            continue
        
        strategy_files = list(strategy_type_dir.glob("*.parquet"))
        
        if not strategy_files:
            print(f"    ❌ No strategy files in {strategy_type}")
            continue
        
        # Sample strategies from this type
        if len(strategy_files) <= samples_per_type:
            selected_files = strategy_files
        else:
            # Take evenly spaced samples
            step = len(strategy_files) // samples_per_type
            selected_files = strategy_files[::step][:samples_per_type]
        
        print(f"    Selected {len(selected_files)}/{len(strategy_files)} strategies")
        
        try:
            start_time = time.time()
            
            # Analyze selected strategies
            type_results = analyzer.analyze_multiple_strategies(
                selected_files,
                classifier_name,
                cost_config
            )
            
            elapsed = time.time() - start_time
            success_count = len(type_results.get('strategies', {}))
            
            print(f"    ✅ {success_count}/{len(selected_files)} strategies analyzed in {elapsed:.1f}s")
            
            # Store results with metadata
            batch_results[strategy_type] = {
                'analysis_results': type_results,
                'metadata': {
                    'strategy_type': strategy_type,
                    'total_available': len(strategy_files),
                    'strategies_analyzed': len(selected_files),
                    'successful_analyses': success_count,
                    'analysis_time': elapsed,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            batch_results[strategy_type] = {
                'analysis_results': None,
                'metadata': {
                    'strategy_type': strategy_type,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    return batch_results


def find_regime_champions_from_batches(combined_results: dict, main_regimes: list):
    """Find regime champions from combined batch results."""
    
    print(f"\n{'='*80}")
    print("FINDING REGIME CHAMPIONS FROM ALL BATCHES")
    print("="*80)
    
    regime_champions = {}
    
    for regime in main_regimes:
        print(f"\n🏆 SEARCHING FOR {regime.upper()} CHAMPION...")
        
        all_candidates = []
        
        # Collect candidates from all strategy types
        for strategy_type, type_data in combined_results.items():
            analysis_results = type_data.get('analysis_results')
            if not analysis_results:
                continue
                
            for strategy_name, strategy_data in analysis_results.get('strategies', {}).items():
                regime_perf = strategy_data.get('regime_performance', {}).get(regime)
                
                if regime_perf and regime_perf['trade_count'] >= 30:  # Min trade threshold
                    candidate = {
                        'strategy_name': strategy_name,
                        'strategy_type': strategy_type,
                        'return': regime_perf['net_percentage_return'],
                        'log_return': regime_perf['total_net_log_return'],
                        'trades': regime_perf['trade_count'],
                        'win_rate': regime_perf['win_rate'],
                        'avg_trade_return': regime_perf['avg_trade_return'],
                        'profit_factor': regime_perf.get('profit_factor', 0),
                        'avg_bars_held': regime_perf['avg_bars_held']
                    }
                    all_candidates.append(candidate)
        
        if not all_candidates:
            print(f"  ❌ No qualified candidates for {regime}")
            regime_champions[regime] = None
            continue
        
        # Sort by return
        all_candidates.sort(key=lambda x: x['return'], reverse=True)
        
        champion = all_candidates[0]
        top_10 = all_candidates[:10]
        
        print(f"  🥇 CHAMPION: {champion['strategy_name']} ({champion['strategy_type']})")
        print(f"     Return: {champion['return']:.2%}")
        print(f"     Trades: {champion['trades']}")
        print(f"     Win Rate: {champion['win_rate']:.1%}")
        print(f"     Profit Factor: {champion['profit_factor']:.2f}")
        
        print(f"  🏅 Top 10 contenders:")
        for i, candidate in enumerate(top_10):
            print(f"     {i+1:2d}. {candidate['strategy_name']:<40} ({candidate['strategy_type']:<25}): {candidate['return']:>8.2%}")
        
        regime_champions[regime] = {
            'champion': champion,
            'top_10': top_10,
            'total_candidates': len(all_candidates)
        }
    
    return regime_champions


def build_final_ensemble(regime_champions: dict, combined_results: dict):
    """Build final ensemble configuration."""
    
    print(f"\n{'='*80}")
    print("BUILDING FINAL REGIME-ADAPTIVE ENSEMBLE")
    print("="*80)
    
    ensemble = {
        'classifier': 'SPY_market_regime_grid_0006_12',
        'ensemble_method': 'regime_specialist',
        'creation_date': pd.Timestamp.now().isoformat(),
        'regimes': {},
        'analysis_summary': {
            'strategy_types_analyzed': len(combined_results),
            'total_strategies_tested': sum(
                data.get('metadata', {}).get('strategies_analyzed', 0) 
                for data in combined_results.values()
            )
        }
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
        print(f"  Total candidates evaluated: {regime_data['total_candidates']}")
    
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
    """Main batched analysis workflow."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    output_dir = workspace_path / "batch_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    print("BATCHED REGIME-ADAPTIVE ENSEMBLE ANALYSIS")
    print("="*60)
    
    # Survey strategy universe
    signals_dir = workspace_path / "traces" / "SPY_1m" / "signals"
    all_strategy_types = [d.name for d in signals_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(all_strategy_types)} strategy types to analyze")
    
    # Configuration
    classifier_name = "SPY_market_regime_grid_0006_12"
    cost_config = ExecutionCostConfig(cost_multiplier=0.99)
    main_regimes = ['bull_ranging', 'bear_ranging', 'neutral']
    
    # Process in batches of 5 strategy types
    batch_size = 5
    batches = [all_strategy_types[i:i+batch_size] for i in range(0, len(all_strategy_types), batch_size)]
    
    print(f"Processing {len(batches)} batches of {batch_size} strategy types each")
    print(f"Using classifier: {classifier_name}")
    print(f"Execution cost: 1% multiplier")
    
    # Process each batch
    for i, batch_types in enumerate(batches, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING BATCH {i}/{len(batches)}")
        print(f"Strategy types: {batch_types}")
        print("="*60)
        
        batch_results = analyze_strategy_type_batch(
            batch_types,
            workspace_path,
            classifier_name,
            cost_config,
            samples_per_type=2,  # 2 strategies per type for speed
            batch_id=i
        )
        
        # Save batch results
        batch_file = save_batch_results(batch_results, i, output_dir)
        print(f"  ✅ Batch {i} results saved: {batch_file.name}")
    
    # Combine all batch results
    print(f"\n{'='*60}")
    print("COMBINING ALL BATCH RESULTS")
    print("="*60)
    
    combined_results = load_all_batch_results(output_dir)
    successful_types = len([r for r in combined_results.values() if r.get('analysis_results')])
    
    print(f"✅ Combined results from {successful_types}/{len(all_strategy_types)} strategy types")
    
    # Find regime champions
    regime_champions = find_regime_champions_from_batches(combined_results, main_regimes)
    
    # Build final ensemble
    final_ensemble = build_final_ensemble(regime_champions, combined_results)
    
    # Save comprehensive results
    final_results = {
        'analysis_metadata': {
            'classifier_name': classifier_name,
            'strategy_types_total': len(all_strategy_types),
            'strategy_types_analyzed': successful_types,
            'cost_config': cost_config.__dict__,
            'regimes_analyzed': main_regimes,
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'regime_champions': regime_champions,
        'final_ensemble': final_ensemble,
        'detailed_results_location': str(output_dir)
    }
    
    # Save final ensemble
    final_file = workspace_path / "comprehensive_regime_ensemble.json"
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Save just the ensemble config
    ensemble_config_file = workspace_path / "final_regime_ensemble_config.json"
    with open(ensemble_config_file, 'w') as f:
        json.dump(final_ensemble, f, indent=2, default=str)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"  📊 Full results: {final_file}")
    print(f"  ⚙️  Ensemble config: {ensemble_config_file}")
    print(f"  📁 Batch details: {output_dir}")
    
    print(f"\n🏆 FINAL REGIME CHAMPIONS:")
    for regime, regime_data in regime_champions.items():
        if regime_data:
            champion = regime_data['champion']
            print(f"  {regime.upper()}: {champion['strategy_name']} ({champion['strategy_type']}) - {champion['return']:.2%}")
        else:
            print(f"  {regime.upper()}: No champion found")
    
    print(f"\n🎯 Expected ensemble return: {final_ensemble['expected_performance']['total_expected_return']:.2%}")
    
    return final_ensemble


if __name__ == "__main__":
    try:
        ensemble = main()
        print(f"\n🚀 REGIME-ADAPTIVE ENSEMBLE READY FOR IMPLEMENTATION!")
        
    except Exception as e:
        print(f"\n❌ Batched ensemble analysis failed: {e}")
        import traceback
        traceback.print_exc()