#!/usr/bin/env python3
"""
Analyze classifier regime distributions in the complete_strategy_grid_v1_acf6c935 workspace
to check if distributions are more balanced after code improvements.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import json

def reconstruct_sparse_series(data: pd.DataFrame, column_name: str, total_length: int) -> pd.Series:
    """
    Reconstruct a full sparse series from stored sparse data.
    Uses forward-fill to propagate values between sparse updates.
    """
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
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        if df.empty:
            return None
            
        total_length = len(df)
        
        # Look for regime-related columns
        regime_columns = [col for col in df.columns if 'regime' in col.lower() or 'state' in col.lower()]
        
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
    
    # Results organized by classifier type
    classifier_results = defaultdict(list)
    
    # Process each classifier type directory
    for classifier_dir in classifiers_path.iterdir():
        if not classifier_dir.is_dir():
            continue
            
        classifier_type = classifier_dir.name
        print(f"\nAnalyzing classifier type: {classifier_type}")
        
        # Process all parquet files in this classifier type
        parquet_files = list(classifier_dir.glob("*.parquet"))
        print(f"Found {len(parquet_files)} files")
        
        for file_path in parquet_files:
            result = analyze_classifier_file(file_path)
            if result:
                # Extract parameters from filename
                filename = file_path.stem
                result['filename'] = filename
                result['classifier_type'] = classifier_type
                classifier_results[classifier_type].append(result)
    
    return dict(classifier_results)

def create_summary_report(results: dict):
    """Create a comprehensive summary report of regime distributions."""
    print("\n" + "="*80)
    print("CLASSIFIER REGIME DISTRIBUTION ANALYSIS")
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
                
                # Track unique regimes
                for regime in stats['regime_counts'].keys():
                    if regime is not None:
                        type_stats['regimes_found'].add(str(regime))
                
                print(f"  File: {file_data['filename']}")
                print(f"    Column: {regime_col}")
                print(f"    Total bars: {stats['total_bars']:,}")
                print(f"    Valid regime bars: {stats['valid_regime_bars']:,}")
                print(f"    Unique regimes: {stats['unique_regimes']}")
                print(f"    Regime distribution: {stats['regime_percentages']}")
                print(f"    Entropy: {stats['entropy']:.3f}")
                print(f"    Balance ratio (min/max): {stats['balance_ratio']:.3f}")
                print(f"    Max regime %: {stats['max_regime_pct']:.1f}%")
                print()
        
        # Calculate averages for this classifier type
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
        print(f"    Unique regimes found: {type_stats['regimes_found']}")
    
    return overall_stats

def create_balance_assessment(overall_stats: list):
    """Assess which classifiers are most/least balanced."""
    print("\n" + "="*80)
    print("BALANCE ASSESSMENT")
    print("="*80)
    
    # Sort by balance metrics
    by_entropy = sorted(overall_stats, key=lambda x: x['avg_entropy'], reverse=True)
    by_balance_ratio = sorted(overall_stats, key=lambda x: x['avg_balance_ratio'], reverse=True)
    by_max_regime = sorted(overall_stats, key=lambda x: x['avg_max_regime_pct'])
    
    print("\nMOST BALANCED (by entropy - higher is better):")
    print("-" * 50)
    for i, stats in enumerate(by_entropy[:5]):
        print(f"{i+1}. {stats['classifier_type']}")
        print(f"   Entropy: {stats['avg_entropy']:.3f}")
        print(f"   Balance ratio: {stats['avg_balance_ratio']:.3f}")
        print(f"   Max regime %: {stats['avg_max_regime_pct']:.1f}%")
        print()
    
    print("\nMOST BALANCED (by balance ratio - higher is better):")
    print("-" * 50)
    for i, stats in enumerate(by_balance_ratio[:5]):
        print(f"{i+1}. {stats['classifier_type']}")
        print(f"   Balance ratio: {stats['avg_balance_ratio']:.3f}")
        print(f"   Entropy: {stats['avg_entropy']:.3f}")
        print(f"   Max regime %: {stats['avg_max_regime_pct']:.1f}%")
        print()
    
    print("\nLEAST BALANCED (highest max regime %):")
    print("-" * 50)
    for i, stats in enumerate(reversed(by_max_regime[-5:])):
        print(f"{i+1}. {stats['classifier_type']}")
        print(f"   Max regime %: {stats['avg_max_regime_pct']:.1f}%")
        print(f"   Balance ratio: {stats['avg_balance_ratio']:.3f}")
        print(f"   Entropy: {stats['avg_entropy']:.3f}")
        print()
    
    # Balance quality assessment
    print("\nBALANCE QUALITY ASSESSMENT:")
    print("-" * 50)
    
    excellent = []
    good = []
    fair = []
    poor = []
    
    for stats in overall_stats:
        if stats['avg_balance_ratio'] >= 0.8 and stats['avg_entropy'] >= 1.5:
            excellent.append(stats['classifier_type'])
        elif stats['avg_balance_ratio'] >= 0.6 and stats['avg_entropy'] >= 1.2:
            good.append(stats['classifier_type'])
        elif stats['avg_balance_ratio'] >= 0.4 and stats['avg_entropy'] >= 0.8:
            fair.append(stats['classifier_type'])
        else:
            poor.append(stats['classifier_type'])
    
    print(f"EXCELLENT balance (ratio ≥ 0.8, entropy ≥ 1.5): {len(excellent)}")
    for clf in excellent:
        print(f"  - {clf}")
    
    print(f"\nGOOD balance (ratio ≥ 0.6, entropy ≥ 1.2): {len(good)}")
    for clf in good:
        print(f"  - {clf}")
    
    print(f"\nFAIR balance (ratio ≥ 0.4, entropy ≥ 0.8): {len(fair)}")
    for clf in fair:
        print(f"  - {clf}")
    
    print(f"\nPOOR balance (below fair thresholds): {len(poor)}")
    for clf in poor:
        print(f"  - {clf}")

def main():
    workspace_path = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_acf6c935"
    
    print("Analyzing classifier regime distributions...")
    print(f"Workspace: {workspace_path}")
    
    # Analyze all classifiers
    results = analyze_workspace_classifiers(workspace_path)
    
    if not results:
        print("No classifier results found!")
        return
    
    # Create summary report
    overall_stats = create_summary_report(results)
    
    # Create balance assessment
    create_balance_assessment(overall_stats)
    
    # Save detailed results
    output_file = "/Users/daws/ADMF-PC/classifier_regime_analysis.json"
    with open(output_file, 'w') as f:
        # Convert sets to lists for JSON serialization
        json_stats = []
        for stats in overall_stats:
            stats_copy = stats.copy()
            if isinstance(stats_copy['regimes_found'], set):
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