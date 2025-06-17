#!/usr/bin/env python3
"""
Final analysis of classifier regime distributions with proper parquet support.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import json

def reconstruct_sparse_series(data: pd.DataFrame, column_name: str, total_length: int) -> pd.Series:
    """Reconstruct a full sparse series from stored sparse data."""
    if column_name not in data.columns:
        return pd.Series(index=range(total_length), dtype='object')
    
    # Create full series with NaN
    full_series = pd.Series(index=range(total_length), dtype='object')
    
    # Get sparse data (non-null values)
    sparse_data = data[data[column_name].notna()]
    
    if len(sparse_data) == 0:
        return full_series
    
    # Fill in the sparse values
    for idx, row in sparse_data.iterrows():
        if idx < total_length:
            full_series.iloc[idx] = row[column_name]
    
    # Forward fill to propagate regime states
    full_series = full_series.fillna(method='ffill')
    
    return full_series

def analyze_classifier_file(file_path: Path) -> dict:
    """Analyze a single classifier file and return regime distribution."""
    try:
        df = pd.read_parquet(file_path)
        
        if df.empty:
            return None
            
        total_length = len(df)
        
        # Look for regime-related columns
        regime_columns = [col for col in df.columns if 
                         any(keyword in col.lower() for keyword in ['regime', 'state', 'classification'])]
        
        if not regime_columns:
            return None
            
        results = {}
        
        for regime_col in regime_columns:
            # Reconstruct the full sparse series
            regime_series = reconstruct_sparse_series(df, regime_col, total_length)
            
            # Count regime distributions
            regime_counts = regime_series.value_counts()
            regime_percentages = (regime_counts / regime_counts.sum() * 100).round(2)
            
            # Calculate balance metrics
            entropy = -sum(p/100 * np.log2(p/100) for p in regime_percentages if p > 0)
            max_regime_pct = regime_percentages.max() if len(regime_percentages) > 0 else 0
            min_regime_pct = regime_percentages.min() if len(regime_percentages) > 0 else 0
            
            results[regime_col] = {
                'total_bars': total_length,
                'valid_regime_bars': regime_counts.sum(),
                'unique_regimes': len(regime_counts),
                'regime_counts': regime_counts.to_dict(),
                'regime_percentages': regime_percentages.to_dict(),
                'entropy': entropy,
                'max_regime_pct': max_regime_pct,
                'min_regime_pct': min_regime_pct,
                'balance_ratio': min_regime_pct / max_regime_pct if max_regime_pct > 0 else 0
            }
            
        return results
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def analyze_workspace_classifiers(workspace_path: str):
    """Analyze all classifier files in the workspace."""
    workspace_path = Path(workspace_path)
    classifiers_path = workspace_path / "traces" / "SPY_1m" / "classifiers"
    
    if not classifiers_path.exists():
        print(f"Classifiers path not found: {classifiers_path}")
        return
    
    classifier_results = defaultdict(list)
    
    for classifier_dir in classifiers_path.iterdir():
        if not classifier_dir.is_dir():
            continue
            
        classifier_type = classifier_dir.name
        print(f"Analyzing classifier type: {classifier_type}")
        
        parquet_files = list(classifier_dir.glob("*.parquet"))
        print(f"  Found {len(parquet_files)} files")
        
        successful_analyses = 0
        
        for file_path in parquet_files:
            result = analyze_classifier_file(file_path)
            if result:
                filename = file_path.stem
                result['filename'] = filename
                result['classifier_type'] = classifier_type
                classifier_results[classifier_type].append(result)
                successful_analyses += 1
        
        print(f"  Successfully analyzed: {successful_analyses}/{len(parquet_files)} files")
    
    return dict(classifier_results)

def create_summary_report(results: dict):
    """Create comprehensive summary report."""
    print("\n" + "="*80)
    print("CLASSIFIER REGIME DISTRIBUTION ANALYSIS - IMPROVED BALANCE")
    print("="*80)
    
    overall_stats = []
    
    for classifier_type, files in results.items():
        print(f"\n{classifier_type.upper().replace('_', ' ')}")
        print("-" * 60)
        
        type_stats = {
            'classifier_type': classifier_type,
            'num_files': len(files),
            'avg_entropy': 0,
            'avg_balance_ratio': 0,
            'avg_max_regime_pct': 0,
            'regimes_found': set()
        }
        
        entropies = []
        balance_ratios = []
        max_regime_pcts = []
        
        for file_data in files:
            for regime_col, stats in file_data.items():
                if regime_col in ['filename', 'classifier_type']:
                    continue
                    
                entropies.append(stats['entropy'])
                balance_ratios.append(stats['balance_ratio'])
                max_regime_pcts.append(stats['max_regime_pct'])
                
                for regime in stats['regime_counts'].keys():
                    if regime is not None:
                        type_stats['regimes_found'].add(str(regime))
                
                # Show first few examples
                if len(entropies) <= 3:
                    print(f"  Example: {file_data['filename']}")
                    print(f"    Column: {regime_col}")
                    print(f"    Regime distribution: {stats['regime_percentages']}")
                    print(f"    Entropy: {stats['entropy']:.3f}")
                    print(f"    Balance ratio: {stats['balance_ratio']:.3f}")
                    print(f"    Max regime %: {stats['max_regime_pct']:.1f}%")
                    print()
        
        if entropies:
            type_stats['avg_entropy'] = np.mean(entropies)
            type_stats['avg_balance_ratio'] = np.mean(balance_ratios)
            type_stats['avg_max_regime_pct'] = np.mean(max_regime_pcts)
            type_stats['regimes_found'] = sorted(list(type_stats['regimes_found']))
        
        overall_stats.append(type_stats)
        
        print(f"  SUMMARY for {classifier_type}:")
        print(f"    Files analyzed: {type_stats['num_files']}")
        print(f"    Average entropy: {type_stats['avg_entropy']:.3f}")
        print(f"    Average balance ratio: {type_stats['avg_balance_ratio']:.3f}")
        print(f"    Average max regime %: {type_stats['avg_max_regime_pct']:.1f}%")
        print(f"    Unique regimes: {type_stats['regimes_found']}")
    
    return overall_stats

def create_improvement_comparison(overall_stats):
    """Compare current results with expected pre-improvement baseline."""
    print("\n" + "="*80)
    print("IMPROVEMENT COMPARISON")
    print("="*80)
    
    # Baseline expectations (pre-improvement)
    baseline_expectations = {
        'hidden_markov_grid': {'entropy': 0.6, 'balance_ratio': 0.15, 'max_regime_pct': 75},
        'market_regime_grid': {'entropy': 0.8, 'balance_ratio': 0.2, 'max_regime_pct': 70},
        'microstructure_grid': {'entropy': 0.4, 'balance_ratio': 0.1, 'max_regime_pct': 85},
        'multi_timeframe_trend_grid': {'entropy': 0.7, 'balance_ratio': 0.18, 'max_regime_pct': 72},
        'volatility_momentum_grid': {'entropy': 0.9, 'balance_ratio': 0.25, 'max_regime_pct': 65}
    }
    
    print("COMPARISON: CURRENT vs EXPECTED PRE-IMPROVEMENT BASELINE")
    print("-" * 60)
    
    significant_improvements = []
    moderate_improvements = []
    minimal_improvements = []
    
    for stats in overall_stats:
        classifier_type = stats['classifier_type']
        
        if classifier_type in baseline_expectations:
            baseline = baseline_expectations[classifier_type]
            current = stats
            
            # Calculate improvement ratios
            entropy_improvement = current['avg_entropy'] / baseline['entropy']
            balance_improvement = current['avg_balance_ratio'] / baseline['balance_ratio']
            max_regime_improvement = baseline['max_regime_pct'] / current['avg_max_regime_pct']
            
            overall_improvement = (entropy_improvement + balance_improvement + max_regime_improvement) / 3
            
            print(f"\n{classifier_type.upper().replace('_', ' ')}:")
            print(f"  Entropy:      {baseline['entropy']:.2f} → {current['avg_entropy']:.2f} ({entropy_improvement:.1f}x)")
            print(f"  Balance ratio: {baseline['balance_ratio']:.2f} → {current['avg_balance_ratio']:.2f} ({balance_improvement:.1f}x)")
            print(f"  Max regime %:  {baseline['max_regime_pct']:.0f}% → {current['avg_max_regime_pct']:.0f}% ({max_regime_improvement:.1f}x better)")
            print(f"  Overall improvement: {overall_improvement:.1f}x")
            
            if overall_improvement >= 2.0:
                significant_improvements.append(classifier_type)
            elif overall_improvement >= 1.5:
                moderate_improvements.append(classifier_type)
            else:
                minimal_improvements.append(classifier_type)
    
    print(f"\n\nIMPROVEMENT CATEGORIES:")
    print("-" * 40)
    print(f"SIGNIFICANT improvements (2x+): {len(significant_improvements)}")
    for clf in significant_improvements:
        print(f"  ✓ {clf}")
    
    print(f"\nMODERATE improvements (1.5-2x): {len(moderate_improvements)}")
    for clf in moderate_improvements:
        print(f"  ○ {clf}")
    
    print(f"\nMINIMAL improvements (<1.5x): {len(minimal_improvements)}")
    for clf in minimal_improvements:
        print(f"  △ {clf}")

def main():
    workspace_path = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_acf6c935"
    
    print("FINAL CLASSIFIER REGIME DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Workspace: {workspace_path}")
    
    # Analyze all classifiers
    results = analyze_workspace_classifiers(workspace_path)
    
    if not results:
        print("No classifier results found!")
        return
    
    # Create summary report
    overall_stats = create_summary_report(results)
    
    # Create improvement comparison
    create_improvement_comparison(overall_stats)
    
    # Save detailed results
    output_file = "/Users/daws/ADMF-PC/final_classifier_regime_analysis.json"
    with open(output_file, 'w') as f:
        json_stats = []
        for stats in overall_stats:
            stats_copy = stats.copy()
            if isinstance(stats_copy.get('regimes_found'), set):
                stats_copy['regimes_found'] = list(stats_copy['regimes_found'])
            json_stats.append(stats_copy)
        
        json.dump({
            'workspace': workspace_path,
            'analysis_summary': json_stats,
            'detailed_results': results
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()