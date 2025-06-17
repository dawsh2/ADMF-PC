#!/usr/bin/env python3
"""
Corrected classifier analysis that properly calculates state durations from sparse changes.

Classifiers only broadcast state changes, so we calculate duration by:
- State change BULLISH at bar 100 (start tracking)  
- State change BEARISH at bar 150 → BULLISH lasted 50 bars
- State change SIDEWAYS at bar 175 → BEARISH lasted 25 bars
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def calculate_classifier_state_durations(workspace_path):
    """Calculate actual time spent in each classifier state from sparse change data."""
    
    classifier_dir = workspace_path / "traces" / "SPY_1m" / "classifiers"
    
    if not classifier_dir.exists():
        print(f"Classifier directory not found: {classifier_dir}")
        return {}
    
    classifier_analysis = {}
    classifier_files = list(classifier_dir.rglob("*.parquet"))
    
    print(f"Found {len(classifier_files)} classifier files")
    
    for file_path in classifier_files:
        classifier_name = file_path.stem
        
        try:
            # Read classifier state changes (sparse data)
            df = pd.read_parquet(file_path)
            df = df.rename(columns={'idx': 'bar_idx', 'val': 'state'})
            df = df.sort_values('bar_idx')
            
            if len(df) == 0:
                continue
            
            # Calculate duration for each state by looking at consecutive changes
            state_durations = {}
            total_bars = 0
            
            for i in range(len(df)):
                current_state = df.iloc[i]['state']
                current_bar = df.iloc[i]['bar_idx']
                
                if i < len(df) - 1:
                    # Duration until next state change
                    next_bar = df.iloc[i + 1]['bar_idx']
                    duration = next_bar - current_bar
                else:
                    # Last state - need to estimate duration to end of data
                    # For now, assume it continues for reasonable period
                    duration = 100  # Could be refined with actual data end
                
                if current_state not in state_durations:
                    state_durations[current_state] = 0
                state_durations[current_state] += duration
                total_bars += duration
            
            # Calculate percentages and balance metrics
            states = list(state_durations.keys())
            durations = list(state_durations.values())
            percentages = [(dur / total_bars) * 100 for dur in durations]
            
            num_states = len(states)
            ideal_percentage = 100.0 / num_states
            
            # Balance score: sum of absolute deviations from ideal
            balance_score = sum(abs(pct - ideal_percentage) for pct in percentages)
            
            # Entropy-based balance 
            normalized_probs = [pct/100.0 for pct in percentages]
            entropy = -sum(p * np.log(p) for p in normalized_probs if p > 0)
            max_entropy = np.log(num_states)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            classifier_analysis[classifier_name] = {
                'num_states': num_states,
                'states': states,
                'durations': durations,
                'total_bars': total_bars,
                'percentages': percentages,
                'balance_score': balance_score,
                'normalized_entropy': normalized_entropy,
                'ideal_percentage': ideal_percentage,
                'min_percentage': min(percentages),
                'max_percentage': max(percentages),
                'file_path': str(file_path),
                'num_changes': len(df)
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return classifier_analysis

def print_corrected_analysis(analysis):
    """Print corrected classifier analysis showing actual time spent in states."""
    
    print("\n" + "="*80)
    print("CORRECTED CLASSIFIER STATE DURATION ANALYSIS")
    print("="*80)
    print("Based on sparse state changes to calculate actual time in each regime")
    print("Duration = bars between consecutive state changes")
    print("="*80)
    
    # Sort by balance score (best first)
    sorted_classifiers = sorted(analysis.items(), key=lambda x: x[1]['balance_score'])
    
    print(f"\nAnalyzed {len(sorted_classifiers)} classifiers:")
    print(f"{'Classifier':<40} {'States':<8} {'Changes':<8} {'Balance':<10} {'Entropy':<8} {'Min%':<8} {'Max%':<8}")
    print("-" * 100)
    
    for classifier, stats in sorted_classifiers:
        print(f"{classifier:<40} {stats['num_states']:<8} {stats['num_changes']:<8} "
              f"{stats['balance_score']:<10.2f} {stats['normalized_entropy']:<8.3f} "
              f"{stats['min_percentage']:<8.1f} {stats['max_percentage']:<8.1f}")
    
    # Detailed analysis for top 5
    print(f"\n{'='*80}")
    print("TOP 5 MOST BALANCED CLASSIFIERS (Detailed Duration Analysis)")
    print("="*80)
    
    for i, (classifier, stats) in enumerate(sorted_classifiers[:5]):
        print(f"\n{i+1}. {classifier}")
        print(f"   Total bars covered: {stats['total_bars']:,}")
        print(f"   State changes recorded: {stats['num_changes']}")
        print(f"   Balance Score: {stats['balance_score']:.2f}")
        print(f"   State Duration Distribution:")
        
        for j, (state, duration, pct) in enumerate(zip(stats['states'], stats['durations'], stats['percentages'])):
            deviation = abs(pct - stats['ideal_percentage'])
            print(f"     {state}: {duration:,} bars ({pct:.1f}%, deviation: {deviation:.1f}%)")
    
    return sorted_classifiers

def main():
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    if not workspace_path.exists():
        print(f"Workspace not found: {workspace_path}")
        return
    
    print("CORRECTED CLASSIFIER DURATION ANALYSIS")
    print("="*50)
    print("Calculating actual time spent in each regime state...")
    
    # Analyze classifier state durations
    analysis = calculate_classifier_state_durations(workspace_path)
    
    if not analysis:
        print("No classifier data to analyze")
        return
    
    # Print corrected analysis
    sorted_classifiers = print_corrected_analysis(analysis)
    
    # Save corrected results
    output_file = workspace_path / "corrected_classifier_durations.json"
    
    # Convert for JSON serialization
    json_analysis = {}
    for name, data in analysis.items():
        json_analysis[name] = {k: v for k, v in data.items() if k != 'file_path'}
    
    with open(output_file, 'w') as f:
        json.dump(json_analysis, f, indent=2, default=str)
    
    print(f"\n✓ Corrected analysis saved to: {output_file}")
    
    # Return top classifiers for further use
    return [name for name, _ in sorted_classifiers[:6]]

if __name__ == "__main__":
    main()