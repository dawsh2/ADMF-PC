#!/usr/bin/env python3
"""
Run corrected classifier analysis using the new modular analytics framework.

This script demonstrates the proper way to analyze classifier state distributions
from sparse trace data using the analytics module.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analytics.sparse_trace_analysis import ClassifierAnalyzer
from analytics.sparse_trace_analysis.data_validation import validate_workspace_structure


def main():
    """Run corrected classifier analysis."""
    
    workspace_path = Path("workspaces/complete_strategy_grid_v1_fc4cc700")
    
    print("CORRECTED CLASSIFIER ANALYSIS")
    print("="*50)
    print("Using modular analytics framework")
    print("Calculating actual time spent in each regime state from sparse changes")
    print("="*50)
    
    # Validate workspace structure
    print("Validating workspace structure...")
    workspace_validation = validate_workspace_structure(workspace_path)
    
    if not workspace_validation['is_valid']:
        print("❌ Workspace validation failed:")
        for error in workspace_validation['errors']:
            print(f"  - {error}")
        return
    
    print("✅ Workspace structure is valid")
    print(f"  - Signal files: {workspace_validation['stats']['signals_files']}")
    print(f"  - Classifier files: {workspace_validation['stats']['classifier_files']}")
    
    # Initialize classifier analyzer
    analyzer = ClassifierAnalyzer(workspace_path)
    
    # Analyze all classifiers
    print(f"\nAnalyzing classifier state durations...")
    analysis_results = analyzer.analyze_all_classifiers(total_bars=82000)  # Approximate total bars
    
    if not analysis_results:
        print("❌ No classifier data found to analyze")
        return
    
    # Print analysis summary
    analyzer.print_analysis_summary(analysis_results, top_n=10)
    
    # Select balanced classifiers
    print(f"\n{'='*80}")
    print("SELECTING BALANCED CLASSIFIERS")
    print("="*80)
    
    balanced_classifiers = analyzer.select_balanced_classifiers(
        analysis_results,
        min_state_pct=15.0,    # Relaxed criteria
        max_balance_score=60.0, # Relaxed criteria  
        min_entropy=0.7,       # Relaxed criteria
        max_results=10
    )
    
    if balanced_classifiers:
        print(f"Found {len(balanced_classifiers)} classifiers meeting relaxed criteria:")
        print(f"{'Rank':<5} {'Classifier':<45} {'Balance':<10} {'Entropy':<8} {'Min%':<8}")
        print("-" * 80)
        
        for i, (name, analysis) in enumerate(balanced_classifiers):
            print(f"{i+1:<5} {name:<45} {analysis['balance_score']:<10.2f} "
                  f"{analysis['normalized_entropy']:<8.3f} {analysis['min_percentage']:<8.1f}")
        
        # Save top classifiers for strategy analysis
        top_classifier_names = [name for name, _ in balanced_classifiers[:6]]
        
        print(f"\nTop 6 classifiers recommended for strategy analysis:")
        for i, name in enumerate(top_classifier_names):
            print(f"  {i+1}. {name}")
        
        # Save detailed analysis results
        output_file = workspace_path / "corrected_classifier_analysis.json"
        analyzer.save_analysis_results(analysis_results, output_file)
        
        # Save classifier recommendations
        import json
        recommendations = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_classifiers_analyzed': len(analysis_results),
            'selection_criteria': {
                'min_state_percentage': 15.0,
                'max_balance_score': 60.0,
                'min_normalized_entropy': 0.7
            },
            'balanced_classifiers': [
                {
                    'name': name,
                    'rank': i + 1,
                    'balance_score': analysis['balance_score'],
                    'normalized_entropy': analysis['normalized_entropy'],
                    'min_percentage': analysis['min_percentage'],
                    'max_percentage': analysis['max_percentage'],
                    'num_states': analysis['num_states'],
                    'total_bars': analysis['total_bars']
                }
                for i, (name, analysis) in enumerate(balanced_classifiers)
            ],
            'recommended_for_analysis': top_classifier_names
        }
        
        recommendations_file = workspace_path / "corrected_classifier_recommendations.json"
        with open(recommendations_file, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        print(f"\n✅ Recommendations saved to: {recommendations_file}")
        
        return top_classifier_names
        
    else:
        print("❌ No classifiers found meeting even relaxed balance criteria")
        print("All available classifiers have significant regime imbalances")
        
        # Show the best available options anyway
        sorted_classifiers = sorted(
            analysis_results.items(),
            key=lambda x: x[1]['balance_score']
        )
        
        print(f"\nBest available classifiers (by balance score):")
        for i, (name, analysis) in enumerate(sorted_classifiers[:5]):
            print(f"  {i+1}. {name} (Balance: {analysis['balance_score']:.1f}, "
                  f"Min%: {analysis['min_percentage']:.1f})")
        
        return [name for name, _ in sorted_classifiers[:3]]


if __name__ == "__main__":
    import pandas as pd
    
    try:
        top_classifiers = main()
        
        if top_classifiers:
            print(f"\n{'='*80}")
            print("CORRECTED ANALYSIS COMPLETE")
            print("="*80)
            print(f"✅ Analysis completed successfully")
            print(f"✅ Top {len(top_classifiers)} classifiers identified for strategy analysis")
            print(f"✅ Ready for strategy performance analysis by regime")
            print(f"\nNext steps:")
            print(f"  1. Use recommended classifiers for regime attribution")
            print(f"  2. Analyze strategy performance by regime")
            print(f"  3. Build regime-aware portfolio strategies")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()