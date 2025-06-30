#!/usr/bin/env python3
"""
Compare Keltner workspaces and find successful configurations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_workspace_comparison():
    """Compare different Keltner workspace results"""
    
    # Load the detailed results we just generated
    current_results = pd.read_csv('keltner_workspace_detailed_results.csv')
    
    # Analyze current workspace
    print("="*80)
    print("CURRENT WORKSPACE ANALYSIS")
    print("="*80)
    print(f"Workspace: optimize_keltner_with_filters_20250622_102448")
    
    # Summary statistics
    no_stop = current_results[current_results['stop_loss'].isna()]
    
    print(f"\nTotal strategies tested: {len(no_stop)}")
    print(f"Strategies with positive returns: {(no_stop['total_return'] > 0).sum()}")
    print(f"Average return: {no_stop['total_return'].mean()*100:.2f}%")
    print(f"Best return: {no_stop['total_return'].max()*100:.2f}%")
    print(f"Worst return: {no_stop['total_return'].min()*100:.2f}%")
    print(f"Average Sharpe: {no_stop['sharpe'].mean():.2f}")
    print(f"Best Sharpe: {no_stop['sharpe'].max():.2f}")
    
    # Check for any successful Keltner configurations from other analyses
    print("\n" + "="*80)
    print("SEARCHING FOR SUCCESSFUL KELTNER CONFIGURATIONS")
    print("="*80)
    
    # Look for other analysis results
    analysis_files = [
        'keltner_performance_results.csv',
        'keltner_optimization_results.csv',
        'keltner_filter_results.csv',
        'keltner_comprehensive_results.csv',
        'keltner_5m_strategy_guide.md',
        'KELTNER_PERFORMANCE_SUMMARY.md'
    ]
    
    for file in analysis_files:
        if Path(file).exists():
            print(f"\nFound: {file}")
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    if len(df) == 0:
                        print(f"  File is empty")
                        continue
                    return_col = 'total_return' if 'total_return' in df.columns else 'return'
                    positive = df[df[return_col] > 0]
                    if len(positive) > 0:
                        print(f"  Found {len(positive)} strategies with positive returns")
                        top = positive.nlargest(3, return_col)
                        for _, row in top.iterrows():
                            if 'strategy' in row:
                                print(f"    {row['strategy']}: {row[return_col]*100:.2f}%")
                            else:
                                print(f"    {row}")
                except Exception as e:
                    print(f"  Error reading file: {e}")
            elif file.endswith('.md'):
                with open(file, 'r') as f:
                    content = f.read()
                    if 'positive' in content.lower() or 'profit' in content.lower():
                        print(f"  Contains performance information")
    
    # Recommendations
    print("\n" + "="*80)
    print("ANALYSIS AND RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. FILTER CONFIGURATION ISSUES:")
    print("   - The filter syntax 'signal == 0 or ...' always passes when no position")
    print("   - This essentially disables the filter for entries")
    print("   - Should use 'signal != 0 and ...' for entry filters")
    
    print("\n2. PARAMETER RANGES:")
    print("   - Period range (10-50) may be too wide")
    print("   - Multiplier range (1.0-3.0) includes very tight bands")
    print("   - Consider focusing on successful ranges from other analyses")
    
    print("\n3. SUGGESTED IMPROVEMENTS:")
    print("   - Fix filter syntax to properly filter entries")
    print("   - Test with tighter parameter ranges based on successful configs")
    print("   - Add stop loss and profit target parameters")
    print("   - Consider time-of-day filters")
    
    # Look for specific successful parameters
    print("\n4. SUCCESSFUL KELTNER PARAMETERS (from file searches):")
    
    # Search for specific parameter mentions in Python files
    py_files = Path('.').glob('analyze_keltner*.py')
    for py_file in py_files:
        if py_file.name != 'analyze_keltner_workspace_performance.py':
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'period' in content and 'multiplier' in content:
                        # Look for parameter definitions
                        import re
                        period_matches = re.findall(r'period["\']?\s*[:=]\s*(\d+)', content)
                        multiplier_matches = re.findall(r'multiplier["\']?\s*[:=]\s*(\d+\.?\d*)', content)
                        if period_matches or multiplier_matches:
                            print(f"\n   From {py_file.name}:")
                            if period_matches:
                                print(f"     Periods: {list(set(period_matches))}")
                            if multiplier_matches:
                                print(f"     Multipliers: {list(set(multiplier_matches))}")
            except:
                pass

if __name__ == "__main__":
    analyze_workspace_comparison()