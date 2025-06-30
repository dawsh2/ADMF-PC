#!/usr/bin/env python3
"""
Investigate the performance discrepancy between different analyses.
Earlier we found strategies with >1 bps, now seeing 0.4-0.6 bps.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def compare_analysis_results():
    print("=== INVESTIGATING PERFORMANCE DISCREPANCY ===\n")
    
    # 1. Load the filter group analysis (showed good results)
    filter_df = pd.read_csv("keltner_filter_group_analysis.csv")
    
    print("1. FILTER GROUP ANALYSIS RESULTS:")
    print("="*80)
    print("Top performers from filter analysis:")
    top_filter = filter_df.nlargest(10, 'avg_return_bps')
    for _, row in top_filter.iterrows():
        print(f"  {row['signal_count']:>4.0f} signals: {row['avg_return_bps']:>6.2f} bps, "
              f"{row['avg_trades']:.0f} trades, {row['avg_trades']/252:.1f}/day")
    
    # 2. Check specific workspaces
    print("\n2. WORKSPACE ANALYSIS COMPARISON:")
    print("="*80)
    
    workspaces = [
        ("102448", "/Users/daws/ADMF-PC/workspaces/optimize_keltner_with_filters_20250622_102448"),
        ("112210", "/Users/daws/ADMF-PC/configs/optimize_keltner_with_filters/20250622_112210"),
        ("latest", "/Users/daws/ADMF-PC/config/keltner/results/latest")
    ]
    
    for name, path in workspaces:
        if Path(path).exists():
            print(f"\nWorkspace {name}:")
            metadata_path = Path(path) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Count strategies by signal count
                signal_counts = {}
                for comp_name, comp in metadata['components'].items():
                    if comp_name.startswith('SPY_5m_compiled_strategy_'):
                        signals = comp.get('signal_changes', 0)
                        if signals not in signal_counts:
                            signal_counts[signals] = 0
                        signal_counts[signals] += 1
                
                # Show distribution
                for signals in sorted(signal_counts.keys())[:5]:
                    count = signal_counts[signals]
                    print(f"  {signals:>5} signals: {count} strategies")
    
    # 3. The key insight
    print("\n3. KEY INSIGHT - CALCULATION METHOD:")
    print("="*80)
    print("The discrepancy appears to be from:")
    print("- Filter analysis: Used AVERAGE across multiple strategies")
    print("- Workspace analysis: Looked at INDIVIDUAL strategies")
    print("- OHLC analysis: Applied realistic execution costs and stops")
    
    # 4. Find the actual best performers
    print("\n4. RECONCILING THE RESULTS:")
    print("="*80)
    
    # The 47-signal strategy group
    master_regime = filter_df[filter_df['signal_count'] == 47].iloc[0]
    print(f"\nMaster Regime Filter (47 signals):")
    print(f"  Filter analysis: {master_regime['avg_return_bps']:.2f} bps")
    print(f"  Trade count: {master_regime['avg_trades']:.0f} ({master_regime['avg_trades']/252:.2f}/day)")
    print(f"  Problem: Only 0.09 trades/day - not practical!")
    
    # The 303-signal strategy  
    rsi_volume = filter_df[filter_df['signal_count'] == 303].iloc[0]
    print(f"\nRSI/Volume Filter (303 signals):")
    print(f"  Filter analysis: {rsi_volume['avg_return_bps']:.2f} bps")
    print(f"  Trade count: {rsi_volume['avg_trades']:.0f} ({rsi_volume['avg_trades']/252:.2f}/day)")
    
    # The practical strategies
    practical = filter_df[(filter_df['avg_trades'] >= 500) & (filter_df['avg_return_bps'] > 0)]
    print(f"\nPractical strategies (2+ trades/day, positive returns):")
    for _, row in practical.nlargest(5, 'avg_return_bps').iterrows():
        annual = row['avg_return_bps'] * row['avg_trades'] / 100
        print(f"  {row['signal_count']:>4.0f} signals: {row['avg_return_bps']:.2f} bps, "
              f"{row['avg_trades']/252:.1f}/day, {annual:.1f}% annual")
    
    # 5. The reality check
    print("\n5. REALITY CHECK:")
    print("="*80)
    print("When we analyzed with full OHLC data:")
    print("- Applied realistic execution costs (0.5 bps)")
    print("- Tested actual stop losses against high/low")
    print("- Result: 0.45-0.69 bps/trade")
    print("\nThis suggests the filter analysis may have:")
    print("- Used different execution cost assumptions")
    print("- Averaged across strategies (hiding poor performers)")
    print("- Not included realistic market friction")

if __name__ == "__main__":
    compare_analysis_results()