"""
Classifier analysis module for sparse trace data.

Analyzes classifier state distributions by calculating actual time spent in each state
from sparse state change data. Provides balance metrics and selection criteria.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json


def calculate_state_durations(classifier_df: pd.DataFrame, total_bars: Optional[int] = None) -> Dict[str, int]:
    """
    Calculate actual time spent in each classifier state from sparse change data.
    
    Args:
        classifier_df: DataFrame with columns ['bar_idx', 'state']
        total_bars: Total bars in dataset (for estimating final state duration)
        
    Returns:
        Dictionary mapping state names to duration in bars
    """
    if classifier_df.empty:
        return {}
    
    # Ensure sorted by bar index
    classifier_df = classifier_df.sort_values('bar_idx').reset_index(drop=True)
    
    state_durations = {}
    
    for i in range(len(classifier_df)):
        current_state = classifier_df.iloc[i]['state']
        current_bar = classifier_df.iloc[i]['bar_idx']
        
        if i < len(classifier_df) - 1:
            # Duration until next state change
            next_bar = classifier_df.iloc[i + 1]['bar_idx']
            duration = next_bar - current_bar
        else:
            # Last state - estimate duration to end
            if total_bars is not None:
                duration = total_bars - current_bar
            else:
                # Conservative estimate based on average previous durations
                if i > 0:
                    avg_duration = current_bar / i  # Average duration so far
                    duration = min(avg_duration, 1000)  # Cap at reasonable value
                else:
                    duration = 100  # Default estimate
        
        if current_state not in state_durations:
            state_durations[current_state] = 0
        state_durations[current_state] += duration
    
    return state_durations


def analyze_classifier_balance(
    state_durations: Dict[str, int],
    min_state_pct: float = 5.0,
    max_balance_score: float = 50.0
) -> Dict[str, Any]:
    """
    Analyze balance metrics for a classifier's state distribution.
    
    Args:
        state_durations: Dictionary mapping states to duration in bars
        min_state_pct: Minimum percentage for any state (balance criterion)
        max_balance_score: Maximum balance score for "good" balance
        
    Returns:
        Dictionary with balance analysis results
    """
    if not state_durations:
        return {
            'num_states': 0,
            'total_bars': 0,
            'is_balanced': False,
            'balance_score': float('inf'),
            'normalized_entropy': 0.0
        }
    
    # Calculate basic metrics
    states = list(state_durations.keys())
    durations = list(state_durations.values())
    total_bars = sum(durations)
    
    if total_bars == 0:
        return {
            'num_states': len(states),
            'total_bars': 0,
            'is_balanced': False,
            'balance_score': float('inf'),
            'normalized_entropy': 0.0
        }
    
    # Calculate percentages
    percentages = [(duration / total_bars) * 100 for duration in durations]
    
    # Balance metrics
    num_states = len(states)
    ideal_percentage = 100.0 / num_states
    
    # Balance score: sum of absolute deviations from ideal
    balance_score = sum(abs(pct - ideal_percentage) for pct in percentages)
    
    # Normalized entropy
    normalized_probs = [pct / 100.0 for pct in percentages]
    entropy = -sum(p * np.log(p) for p in normalized_probs if p > 0)
    max_entropy = np.log(num_states) if num_states > 1 else 1.0
    normalized_entropy = entropy / max_entropy
    
    # Balance criteria
    min_percentage = min(percentages)
    max_percentage = max(percentages)
    
    is_balanced = (
        balance_score <= max_balance_score and
        min_percentage >= min_state_pct and
        normalized_entropy >= 0.7
    )
    
    return {
        'num_states': num_states,
        'states': states,
        'durations': durations,
        'total_bars': total_bars,
        'percentages': percentages,
        'balance_score': balance_score,
        'normalized_entropy': normalized_entropy,
        'ideal_percentage': ideal_percentage,
        'min_percentage': min_percentage,
        'max_percentage': max_percentage,
        'is_balanced': is_balanced,
        'num_changes': len(states)  # Actually number of unique states
    }


class ClassifierAnalyzer:
    """Main class for analyzing classifier balance and state distributions."""
    
    def __init__(self, workspace_path: Path):
        """
        Initialize classifier analyzer.
        
        Args:
            workspace_path: Path to workspace containing trace data
        """
        self.workspace_path = Path(workspace_path)
        self.classifier_dir = self.workspace_path / "traces" / "SPY_1m" / "classifiers"
        
    def load_classifier_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load classifier data from parquet file.
        
        Args:
            file_path: Path to classifier parquet file
            
        Returns:
            DataFrame with normalized column names, or None if error
        """
        try:
            df = pd.read_parquet(file_path)
            
            # Normalize column names
            df = df.rename(columns={
                'idx': 'bar_idx',
                'val': 'state',
                'px': 'price'
            })
            
            # Ensure required columns exist
            if 'bar_idx' not in df.columns or 'state' not in df.columns:
                print(f"Warning: Missing required columns in {file_path}")
                return None
                
            return df[['bar_idx', 'state']].sort_values('bar_idx')
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def analyze_single_classifier(
        self, 
        file_path: Path,
        total_bars: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a single classifier file.
        
        Args:
            file_path: Path to classifier parquet file
            total_bars: Total bars in dataset for duration estimation
            
        Returns:
            Analysis results dictionary or None if error
        """
        classifier_df = self.load_classifier_data(file_path)
        
        if classifier_df is None or classifier_df.empty:
            return None
        
        # Calculate state durations
        state_durations = calculate_state_durations(classifier_df, total_bars)
        
        # Analyze balance
        balance_analysis = analyze_classifier_balance(state_durations)
        
        # Add metadata
        balance_analysis.update({
            'classifier_name': file_path.stem,
            'file_path': str(file_path),
            'num_changes': len(classifier_df)  # Actual number of state changes
        })
        
        return balance_analysis
    
    def analyze_all_classifiers(
        self, 
        total_bars: Optional[int] = None,
        classifier_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze all classifiers in the workspace.
        
        Args:
            total_bars: Total bars in dataset for duration estimation
            classifier_types: List of classifier types to analyze (None = all)
            
        Returns:
            Dictionary mapping classifier names to analysis results
        """
        if not self.classifier_dir.exists():
            print(f"Classifier directory not found: {self.classifier_dir}")
            return {}
        
        classifier_files = list(self.classifier_dir.rglob("*.parquet"))
        
        if classifier_types:
            # Filter by classifier types
            filtered_files = []
            for file_path in classifier_files:
                for classifier_type in classifier_types:
                    if classifier_type in str(file_path.parent):
                        filtered_files.append(file_path)
                        break
            classifier_files = filtered_files
        
        print(f"Found {len(classifier_files)} classifier files to analyze")
        
        results = {}
        
        for file_path in classifier_files:
            classifier_name = file_path.stem
            analysis = self.analyze_single_classifier(file_path, total_bars)
            
            if analysis is not None:
                results[classifier_name] = analysis
            
        return results
    
    def select_balanced_classifiers(
        self,
        analysis_results: Dict[str, Any],
        min_state_pct: float = 10.0,
        max_balance_score: float = 40.0,
        min_entropy: float = 0.8,
        max_results: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Select the most balanced classifiers based on criteria.
        
        Args:
            analysis_results: Results from analyze_all_classifiers
            min_state_pct: Minimum percentage for any state
            max_balance_score: Maximum balance score
            min_entropy: Minimum normalized entropy
            max_results: Maximum number of classifiers to return
            
        Returns:
            List of (classifier_name, analysis) tuples, sorted by balance score
        """
        # Filter by criteria
        qualified = []
        
        for name, analysis in analysis_results.items():
            if (analysis['min_percentage'] >= min_state_pct and
                analysis['balance_score'] <= max_balance_score and
                analysis['normalized_entropy'] >= min_entropy):
                qualified.append((name, analysis))
        
        # If no classifiers meet strict criteria, relax and take best available
        if not qualified:
            print(f"No classifiers meet strict criteria. Using best available...")
            min_state_pct = max(5.0, min_state_pct * 0.5)
            max_balance_score = min(100.0, max_balance_score * 2)
            min_entropy = max(0.5, min_entropy * 0.8)
            
            for name, analysis in analysis_results.items():
                if (analysis['min_percentage'] >= min_state_pct and
                    analysis['balance_score'] <= max_balance_score and
                    analysis['normalized_entropy'] >= min_entropy):
                    qualified.append((name, analysis))
        
        # Sort by balance score (lower is better)
        qualified.sort(key=lambda x: x[1]['balance_score'])
        
        return qualified[:max_results]
    
    def print_analysis_summary(
        self,
        analysis_results: Dict[str, Any],
        top_n: int = 10
    ) -> None:
        """
        Print a formatted summary of classifier analysis results.
        
        Args:
            analysis_results: Results from analyze_all_classifiers
            top_n: Number of top classifiers to show in detail
        """
        if not analysis_results:
            print("No classifier analysis results to display")
            return
        
        print("\n" + "="*90)
        print("CLASSIFIER STATE DURATION ANALYSIS")
        print("="*90)
        print("Calculated from sparse state changes (actual time in each regime)")
        print("="*90)
        
        # Sort by balance score
        sorted_results = sorted(
            analysis_results.items(),
            key=lambda x: x[1]['balance_score']
        )
        
        # Summary table
        print(f"\nAnalyzed {len(sorted_results)} classifiers:")
        print(f"{'Classifier':<40} {'States':<8} {'Changes':<8} {'Balance':<10} {'Entropy':<8} {'Min%':<8} {'Max%':<8}")
        print("-" * 100)
        
        for name, analysis in sorted_results:
            print(f"{name:<40} {analysis['num_states']:<8} {analysis['num_changes']:<8} "
                  f"{analysis['balance_score']:<10.2f} {analysis['normalized_entropy']:<8.3f} "
                  f"{analysis['min_percentage']:<8.1f} {analysis['max_percentage']:<8.1f}")
        
        # Detailed analysis for top classifiers
        print(f"\n{'='*90}")
        print(f"TOP {top_n} MOST BALANCED CLASSIFIERS (Detailed)")
        print("="*90)
        
        for i, (name, analysis) in enumerate(sorted_results[:top_n]):
            print(f"\n{i+1}. {name}")
            print(f"   Total bars covered: {analysis['total_bars']:,}")
            print(f"   State changes: {analysis['num_changes']}")
            print(f"   Balance Score: {analysis['balance_score']:.2f} (lower = better)")
            print(f"   Normalized Entropy: {analysis['normalized_entropy']:.3f} (higher = better)")
            print(f"   State Duration Distribution:")
            
            for state, duration, pct in zip(analysis['states'], analysis['durations'], analysis['percentages']):
                deviation = abs(pct - analysis['ideal_percentage'])
                print(f"     {state}: {duration:,} bars ({pct:.1f}%, deviation: {deviation:.1f}%)")
    
    def save_analysis_results(
        self,
        analysis_results: Dict[str, Any],
        output_file: Path
    ) -> None:
        """
        Save analysis results to JSON file.
        
        Args:
            analysis_results: Results from analyze_all_classifiers
            output_file: Path to output JSON file
        """
        # Convert for JSON serialization
        json_results = {}
        for name, analysis in analysis_results.items():
            json_results[name] = {
                k: v for k, v in analysis.items() 
                if k not in ['file_path']  # Exclude file paths
            }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"Analysis results saved to: {output_file}")


def load_classifier_analysis_results(file_path: Path) -> Dict[str, Any]:
    """
    Load previously saved classifier analysis results.
    
    Args:
        file_path: Path to JSON file with analysis results
        
    Returns:
        Dictionary with analysis results
    """
    with open(file_path, 'r') as f:
        return json.load(f)