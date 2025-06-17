"""
Regime attribution module for sparse trace analysis.

Handles mapping trades to market regimes based on sparse classifier state changes.
Attributes trades to the regime where position was opened.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def get_regime_at_bar(
    bar_idx: int,
    classifier_changes: pd.DataFrame,
    default_regime: str = 'unknown'
) -> str:
    """
    Determine the active regime at a specific bar index using sparse classifier changes.
    
    Args:
        bar_idx: Bar index to query
        classifier_changes: DataFrame with columns ['bar_idx', 'state'] sorted by bar_idx
        default_regime: Default regime if no data available
        
    Returns:
        Active regime state at the specified bar
    """
    if classifier_changes.empty:
        return default_regime
    
    # Find most recent state change before or at the query bar
    relevant_changes = classifier_changes[classifier_changes['bar_idx'] <= bar_idx]
    
    if len(relevant_changes) == 0:
        return default_regime
    
    # Return the most recent state
    return relevant_changes.iloc[-1]['state']


def attribute_trades_to_regimes(
    trades: List[Dict[str, Any]],
    classifier_changes: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Attribute trades to regimes based on where positions were opened.
    
    Args:
        trades: List of trade dictionaries with 'entry_bar' key
        classifier_changes: DataFrame with sparse classifier state changes
        
    Returns:
        List of trades with 'regime' field added
    """
    attributed_trades = []
    
    for trade in trades:
        entry_bar = trade['entry_bar']
        regime = get_regime_at_bar(entry_bar, classifier_changes)
        
        # Add regime to trade data
        trade_with_regime = trade.copy()
        trade_with_regime['regime'] = regime
        attributed_trades.append(trade_with_regime)
    
    return attributed_trades


def group_trades_by_regime(
    attributed_trades: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group trades by their attributed regime.
    
    Args:
        attributed_trades: List of trades with 'regime' field
        
    Returns:
        Dictionary mapping regime names to lists of trades
    """
    regime_trades = {}
    
    for trade in attributed_trades:
        regime = trade.get('regime', 'unknown')
        
        if regime not in regime_trades:
            regime_trades[regime] = []
        
        regime_trades[regime].append(trade)
    
    return regime_trades


def analyze_regime_transitions(
    classifier_changes: pd.DataFrame,
    min_duration: int = 1
) -> Dict[str, Any]:
    """
    Analyze regime transition patterns from sparse classifier changes.
    
    Args:
        classifier_changes: DataFrame with sparse classifier state changes
        min_duration: Minimum duration to consider a valid regime period
        
    Returns:
        Dictionary with transition analysis results
    """
    if len(classifier_changes) < 2:
        return {
            'num_transitions': 0,
            'transition_matrix': {},
            'average_duration': {},
            'regime_persistence': {}
        }
    
    # Calculate durations and transitions
    transitions = []
    durations = {}
    
    for i in range(len(classifier_changes) - 1):
        current_state = classifier_changes.iloc[i]['state']
        next_state = classifier_changes.iloc[i + 1]['state']
        current_bar = classifier_changes.iloc[i]['bar_idx']
        next_bar = classifier_changes.iloc[i + 1]['bar_idx']
        
        duration = next_bar - current_bar
        
        # Track durations by state
        if current_state not in durations:
            durations[current_state] = []
        durations[current_state].append(duration)
        
        # Track transitions (only if duration meets minimum)
        if duration >= min_duration:
            transitions.append((current_state, next_state, duration))
    
    # Build transition matrix
    transition_matrix = {}
    for from_state, to_state, duration in transitions:
        if from_state not in transition_matrix:
            transition_matrix[from_state] = {}
        if to_state not in transition_matrix[from_state]:
            transition_matrix[from_state][to_state] = 0
        transition_matrix[from_state][to_state] += 1
    
    # Calculate average durations
    average_duration = {}
    for state, state_durations in durations.items():
        average_duration[state] = np.mean(state_durations)
    
    # Calculate regime persistence (probability of staying in same state)
    regime_persistence = {}
    for from_state, transitions_from in transition_matrix.items():
        total_transitions = sum(transitions_from.values())
        self_transitions = transitions_from.get(from_state, 0)
        regime_persistence[from_state] = self_transitions / total_transitions if total_transitions > 0 else 0
    
    return {
        'num_transitions': len(transitions),
        'transition_matrix': transition_matrix,
        'average_duration': average_duration,
        'regime_persistence': regime_persistence,
        'total_regimes': len(set(durations.keys()))
    }


class RegimeAttributor:
    """Main class for regime attribution and analysis."""
    
    def __init__(self, workspace_path: Path):
        """
        Initialize regime attributor.
        
        Args:
            workspace_path: Path to workspace containing trace data
        """
        self.workspace_path = Path(workspace_path)
        self.classifier_dir = self.workspace_path / "traces" / "SPY_1m" / "classifiers"
        self._classifier_cache = {}
    
    def load_classifier_changes(self, classifier_name: str) -> Optional[pd.DataFrame]:
        """
        Load classifier state changes with caching.
        
        Args:
            classifier_name: Name of classifier file (without .parquet extension)
            
        Returns:
            DataFrame with classifier state changes or None if not found
        """
        if classifier_name in self._classifier_cache:
            return self._classifier_cache[classifier_name]
        
        # Find classifier file
        classifier_files = list(self.classifier_dir.rglob(f"{classifier_name}.parquet"))
        
        if not classifier_files:
            print(f"Classifier file not found: {classifier_name}")
            return None
        
        try:
            df = pd.read_parquet(classifier_files[0])
            
            # Normalize column names
            df = df.rename(columns={
                'idx': 'bar_idx',
                'val': 'state',
                'px': 'price'
            })
            
            # Cache and return
            result = df[['bar_idx', 'state']].sort_values('bar_idx')
            self._classifier_cache[classifier_name] = result
            return result
            
        except Exception as e:
            print(f"Error loading classifier {classifier_name}: {e}")
            return None
    
    def attribute_strategy_trades(
        self,
        trades: List[Dict[str, Any]],
        classifier_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Attribute trades to regimes using specified classifier.
        
        Args:
            trades: List of trade dictionaries
            classifier_name: Name of classifier to use for attribution
            
        Returns:
            List of trades with regime attribution or None if error
        """
        classifier_changes = self.load_classifier_changes(classifier_name)
        
        if classifier_changes is None:
            return None
        
        return attribute_trades_to_regimes(trades, classifier_changes)
    
    def analyze_regime_performance(
        self,
        attributed_trades: List[Dict[str, Any]],
        performance_metric: str = 'net_log_return'
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance metrics by regime.
        
        Args:
            attributed_trades: List of trades with regime attribution
            performance_metric: Metric to analyze ('net_log_return', 'net_pnl', etc.)
            
        Returns:
            Dictionary mapping regimes to performance statistics
        """
        regime_trades = group_trades_by_regime(attributed_trades)
        regime_performance = {}
        
        for regime, trades in regime_trades.items():
            if not trades:
                continue
            
            # Extract performance metric values
            values = [trade.get(performance_metric, 0) for trade in trades]
            
            # Calculate statistics
            regime_performance[regime] = {
                'trade_count': len(trades),
                'total_return': sum(values),
                'average_return': np.mean(values),
                'median_return': np.median(values),
                'std_return': np.std(values),
                'win_rate': len([v for v in values if v > 0]) / len(values),
                'best_trade': max(values),
                'worst_trade': min(values),
                'total_bars_held': sum(trade.get('bars_held', 0) for trade in trades),
                'avg_bars_held': np.mean([trade.get('bars_held', 0) for trade in trades])
            }
        
        return regime_performance
    
    def get_regime_summary(
        self,
        classifier_name: str,
        include_transitions: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive regime summary for a classifier.
        
        Args:
            classifier_name: Name of classifier to analyze
            include_transitions: Whether to include transition analysis
            
        Returns:
            Dictionary with regime summary or None if error
        """
        classifier_changes = self.load_classifier_changes(classifier_name)
        
        if classifier_changes is None:
            return None
        
        # Basic regime statistics
        regime_counts = classifier_changes['state'].value_counts()
        unique_regimes = list(regime_counts.index)
        
        summary = {
            'classifier_name': classifier_name,
            'total_changes': len(classifier_changes),
            'unique_regimes': unique_regimes,
            'regime_change_counts': regime_counts.to_dict()
        }
        
        # Add transition analysis if requested
        if include_transitions:
            transition_analysis = analyze_regime_transitions(classifier_changes)
            summary['transition_analysis'] = transition_analysis
        
        return summary
    
    def print_regime_performance_summary(
        self,
        regime_performance: Dict[str, Dict[str, float]],
        classifier_name: str
    ) -> None:
        """
        Print formatted regime performance summary.
        
        Args:
            regime_performance: Results from analyze_regime_performance
            classifier_name: Name of classifier used
        """
        print(f"\n{'='*80}")
        print(f"REGIME PERFORMANCE SUMMARY - {classifier_name}")
        print("="*80)
        print("Trades attributed to regime where position was opened")
        print("="*80)
        
        # Sort regimes by total return
        sorted_regimes = sorted(
            regime_performance.items(),
            key=lambda x: x[1]['total_return'],
            reverse=True
        )
        
        for regime, stats in sorted_regimes:
            print(f"\n{regime.upper()} REGIME:")
            print(f"  Trade Count: {stats['trade_count']}")
            print(f"  Total Return: {stats['total_return']:.4f}")
            print(f"  Average Return: {stats['average_return']:.4f}")
            print(f"  Win Rate: {stats['win_rate']:.2%}")
            print(f"  Best Trade: {stats['best_trade']:.4f}")
            print(f"  Worst Trade: {stats['worst_trade']:.4f}")
            print(f"  Avg Bars Held: {stats['avg_bars_held']:.1f}")
        
        # Overall summary
        total_trades = sum(stats['trade_count'] for stats in regime_performance.values())
        total_return = sum(stats['total_return'] for stats in regime_performance.values())
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Total Return: {total_return:.4f}")
        print(f"  Regimes Analyzed: {len(regime_performance)}")


def validate_regime_attribution(
    attributed_trades: List[Dict[str, Any]],
    classifier_changes: pd.DataFrame
) -> Dict[str, Any]:
    """
    Validate regime attribution results for consistency.
    
    Args:
        attributed_trades: List of trades with regime attribution
        classifier_changes: Original classifier state changes
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_trades': len(attributed_trades),
        'trades_with_regime': 0,
        'unknown_regime_trades': 0,
        'regime_distribution': {},
        'attribution_errors': []
    }
    
    for trade in attributed_trades:
        regime = trade.get('regime', 'unknown')
        entry_bar = trade.get('entry_bar')
        
        # Count regime assignments
        if regime != 'unknown':
            validation_results['trades_with_regime'] += 1
        else:
            validation_results['unknown_regime_trades'] += 1
        
        # Track regime distribution
        if regime not in validation_results['regime_distribution']:
            validation_results['regime_distribution'][regime] = 0
        validation_results['regime_distribution'][regime] += 1
        
        # Validate attribution logic
        if entry_bar is not None and not classifier_changes.empty:
            expected_regime = get_regime_at_bar(entry_bar, classifier_changes)
            if regime != expected_regime:
                validation_results['attribution_errors'].append({
                    'trade_entry_bar': entry_bar,
                    'attributed_regime': regime,
                    'expected_regime': expected_regime
                })
    
    validation_results['attribution_accuracy'] = (
        1.0 - len(validation_results['attribution_errors']) / len(attributed_trades)
        if attributed_trades else 1.0
    )
    
    return validation_results