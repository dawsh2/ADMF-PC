#!/usr/bin/env python3
"""
Analyze classifier state distributions to identify balanced classifiers.

We want classifiers that split the data relatively evenly across their states.
For 3-state classifiers, we ideally want ~33% in each state.
For 2-state classifiers, we ideally want ~50% in each state.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_classifier_distributions(workspace_path):
    """Analyze the distribution of states for each classifier from parquet files."""
    
    classifier_dir = workspace_path / "traces" / "SPY_1m" / "classifiers"
    
    if not classifier_dir.exists():
        print(f"Classifier directory not found: {classifier_dir}")
        return {}
    
    classifier_analysis = {}
    
    # Find all classifier parquet files
    classifier_files = list(classifier_dir.rglob("*.parquet"))
    
    if not classifier_files:
        print("No classifier parquet files found")
        return {}
    
    print(f"Found {len(classifier_files)} classifier files")
    
    for file_path in classifier_files:
        # Extract classifier name from file path
        classifier_name = file_path.stem  # e.g., "SPY_market_regime_grid_0006_12"
        
        try:
            # Read classifier data
            df = pd.read_parquet(file_path)
            
            if 'val' not in df.columns:
                print(f"Warning: No 'val' column in {file_path}")
                continue
            
            # Get state distribution
            state_counts = df['val'].value_counts()
            total_records = len(df)
            
            states = state_counts.index.tolist()
            counts = state_counts.values.tolist()
            percentages = [(count / total_records) * 100 for count in counts]
            
            num_states = len(states)
            ideal_percentage = 100.0 / num_states
            
            # Calculate balance score - lower is better (0 = perfectly balanced)
            # This is the sum of absolute deviations from ideal percentage
            balance_score = sum(abs(pct - ideal_percentage) for pct in percentages)
            
            # Calculate entropy-based balance (higher is better for balanced)
            # Perfect balance gives entropy = log(num_states)
            normalized_probs = [pct/100.0 for pct in percentages]
            entropy = -sum(p * np.log(p) for p in normalized_probs if p > 0)
            max_entropy = np.log(num_states)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            classifier_analysis[classifier_name] = {
                'num_states': num_states,
                'states': states,
                'counts': counts,
                'percentages': percentages,
                'total_records': total_records,
                'balance_score': balance_score,  # Lower is better
                'normalized_entropy': normalized_entropy,  # Higher is better (0-1)
                'ideal_percentage': ideal_percentage,
                'max_deviation': max(abs(pct - ideal_percentage) for pct in percentages),
                'min_percentage': min(percentages),
                'max_percentage': max(percentages),
                'file_path': str(file_path)
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return classifier_analysis

def print_classifier_analysis(analysis):
    """Print detailed analysis of classifier balance."""
    
    print("\n" + "="*80)
    print("CLASSIFIER STATE DISTRIBUTION ANALYSIS")
    print("="*80)
    print("Goal: Find classifiers with balanced state distributions")
    print("- 3-state classifiers should have ~33% in each state")
    print("- 2-state classifiers should have ~50% in each state")
    print("- Balance Score: Lower is better (0 = perfect balance)")
    print("- Normalized Entropy: Higher is better (1 = perfect balance)")
    print("="*80)
    
    # Sort by balance score (best first)
    sorted_classifiers = sorted(analysis.items(), key=lambda x: x[1]['balance_score'])
    
    print(f"\nFound {len(sorted_classifiers)} classifiers:")
    print(f"{'Classifier':<40} {'States':<8} {'Balance':<10} {'Entropy':<8} {'Min%':<8} {'Max%':<8}")
    print("-" * 90)
    
    for classifier, stats in sorted_classifiers:
        print(f"{classifier:<40} {stats['num_states']:<8} {stats['balance_score']:<10.2f} "
              f"{stats['normalized_entropy']:<8.3f} {stats['min_percentage']:<8.1f} "
              f"{stats['max_percentage']:<8.1f}")
    
    # Detailed analysis for top 10 most balanced classifiers
    print(f"\n{'='*80}")
    print("TOP 10 MOST BALANCED CLASSIFIERS (Detailed)")
    print("="*80)
    
    for i, (classifier, stats) in enumerate(sorted_classifiers[:10]):
        print(f"\n{i+1}. {classifier}")
        print(f"   States: {stats['num_states']}")
        print(f"   Balance Score: {stats['balance_score']:.2f} (lower is better)")
        print(f"   Normalized Entropy: {stats['normalized_entropy']:.3f} (higher is better)")
        print(f"   State Distribution:")
        
        for j, (state, pct) in enumerate(zip(stats['states'], stats['percentages'])):
            deviation = abs(pct - stats['ideal_percentage'])
            print(f"     {state}: {pct:.1f}% (deviation: {deviation:.1f}%)")
    
    # Filter recommendations
    print(f"\n{'='*80}")
    print("CLASSIFIER SELECTION RECOMMENDATIONS")
    print("="*80)
    
    # Good balance criteria:
    # - Balance score < 20 (total deviation < 20%)
    # - No state < 20% (avoid extreme skew)
    # - Normalized entropy > 0.85 (good balance)
    
    excellent_classifiers = [
        (name, stats) for name, stats in analysis.items()
        if stats['balance_score'] < 15 and stats['min_percentage'] > 25 and stats['normalized_entropy'] > 0.9
    ]
    
    good_classifiers = [
        (name, stats) for name, stats in analysis.items()
        if stats['balance_score'] < 25 and stats['min_percentage'] > 20 and stats['normalized_entropy'] > 0.85
        and (name, stats) not in excellent_classifiers
    ]
    
    print(f"\nEXCELLENT BALANCE ({len(excellent_classifiers)} classifiers):")
    print("- Balance score < 15")
    print("- All states > 25%") 
    print("- Normalized entropy > 0.9")
    for name, stats in excellent_classifiers:
        print(f"  ✓ {name} (Balance: {stats['balance_score']:.1f}, Entropy: {stats['normalized_entropy']:.3f})")
    
    print(f"\nGOOD BALANCE ({len(good_classifiers)} classifiers):")
    print("- Balance score < 25")
    print("- All states > 20%")
    print("- Normalized entropy > 0.85")
    for name, stats in good_classifiers:
        print(f"  • {name} (Balance: {stats['balance_score']:.1f}, Entropy: {stats['normalized_entropy']:.3f})")
    
    # Poor balance warning
    poor_classifiers = [
        (name, stats) for name, stats in analysis.items()
        if stats['min_percentage'] < 15 or stats['balance_score'] > 50
    ]
    
    if poor_classifiers:
        print(f"\nPOOR BALANCE - AVOID THESE ({len(poor_classifiers)} classifiers):")
        print("- Some state < 15% OR balance score > 50")
        for name, stats in poor_classifiers[:5]:  # Show worst 5
            worst_state_pct = stats['min_percentage']
            print(f"  ✗ {name} (Worst state: {worst_state_pct:.1f}%, Balance: {stats['balance_score']:.1f})")
        if len(poor_classifiers) > 5:
            print(f"  ... and {len(poor_classifiers) - 5} more")
    
    return excellent_classifiers, good_classifiers

def save_classifier_recommendations(excellent, good, output_file):
    """Save classifier recommendations to a file."""
    
    recommendations = {
        'analysis_date': pd.Timestamp.now().isoformat(),
        'selection_criteria': {
            'excellent': {
                'balance_score': '< 15',
                'min_state_percentage': '> 25%',
                'normalized_entropy': '> 0.9'
            },
            'good': {
                'balance_score': '< 25',
                'min_state_percentage': '> 20%', 
                'normalized_entropy': '> 0.85'
            }
        },
        'excellent_classifiers': [name for name, _ in excellent],
        'good_classifiers': [name for name, _ in good],
        'recommended_for_analysis': [name for name, _ in excellent] + [name for name, _ in good]
    }
    
    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"\n✓ Saved classifier recommendations to: {output_file}")
    print(f"  - {len(excellent)} excellent classifiers")
    print(f"  - {len(good)} good classifiers") 
    print(f"  - {len(excellent) + len(good)} total recommended for strategy analysis")

def main():
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    if not workspace_path.exists():
        print(f"Workspace not found: {workspace_path}")
        return
    
    print(f"Analyzing classifier distributions from: {workspace_path}")
    
    # Analyze classifier distributions
    analysis = analyze_classifier_distributions(workspace_path)
    
    if not analysis:
        print("No classifier data to analyze")
        return
    
    # Print analysis results
    excellent, good = print_classifier_analysis(analysis)
    
    # Save recommendations
    output_file = workspace_path / "classifier_recommendations.json"
    save_classifier_recommendations(excellent, good, output_file)

if __name__ == "__main__":
    main()