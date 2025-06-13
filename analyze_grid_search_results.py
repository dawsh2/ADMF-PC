#!/usr/bin/env python3
"""
Grid Search Results Analysis Framework

Processes multi-dimensional results from parameter grid search to find:
1. Best performing strategy configurations
2. Optimal strategy-classifier pairings
3. Regime-specific performance patterns
4. Cross-correlations and feature importance
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import itertools
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class StrategyResult:
    """Results for a single strategy configuration."""
    strategy_type: str
    strategy_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    trades: List[Dict]
    regime_performance: Dict[str, Dict[str, float]]


@dataclass
class ClassifierResult:
    """Results for a single classifier configuration."""
    classifier_type: str
    classifier_id: str
    parameters: Dict[str, Any]
    regime_counts: Dict[str, int]
    regime_durations: Dict[str, float]
    stability_score: float  # How stable are the regimes


class GridSearchAnalyzer:
    """Main analysis class for grid search results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.strategy_results: Dict[str, StrategyResult] = {}
        self.classifier_results: Dict[str, ClassifierResult] = {}
        self.regime_performance_matrix = None
        self.correlation_matrix = None
        
    def load_results(self) -> None:
        """Load all results from the grid search output directory."""
        print("Loading grid search results...")
        
        # Load strategy results
        strategy_files = list(self.results_dir.glob("**/signals_strategy_*.json"))
        for file in strategy_files:
            self._load_strategy_result(file)
            
        # Load classifier results
        classifier_files = list(self.results_dir.glob("**/signals_classifier_*.json"))
        for file in classifier_files:
            self._load_classifier_result(file)
            
        print(f"Loaded {len(self.strategy_results)} strategy configurations")
        print(f"Loaded {len(self.classifier_results)} classifier configurations")
        
    def _load_strategy_result(self, filepath: Path) -> None:
        """Load and parse a single strategy result file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Extract strategy info from metadata
        strategy_params = data.get('metadata', {}).get('strategy_parameters', {})
        
        for strat_id, strat_info in strategy_params.items():
            if 'classifier' not in strat_id:  # Skip classifiers
                # Calculate metrics from signal data
                signals = data.get('changes', [])
                metrics = self._calculate_strategy_metrics(signals, data)
                
                # Get regime performance if available
                regime_perf = self._extract_regime_performance(signals, strat_id)
                
                result = StrategyResult(
                    strategy_type=strat_info.get('type', 'unknown'),
                    strategy_id=strat_id,
                    parameters=strat_info.get('params', {}),
                    metrics=metrics,
                    trades=signals,
                    regime_performance=regime_perf
                )
                
                self.strategy_results[strat_id] = result
                
    def _load_classifier_result(self, filepath: Path) -> None:
        """Load and parse a single classifier result file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Extract classifier info
        classifier_params = data.get('metadata', {}).get('strategy_parameters', {})
        
        for class_id, class_info in classifier_params.items():
            if 'classifier' in class_id:
                # Calculate classifier stability
                changes = data.get('changes', [])
                stability = self._calculate_regime_stability(changes)
                
                # Get regime statistics
                regime_stats = data.get('metadata', {}).get('signal_statistics', {}).get('regime_breakdown', {})
                
                result = ClassifierResult(
                    classifier_type=class_info.get('type', 'unknown'),
                    classifier_id=class_id,
                    parameters=class_info.get('params', {}),
                    regime_counts=regime_stats,
                    regime_durations=self._calculate_regime_durations(changes),
                    stability_score=stability
                )
                
                self.classifier_results[class_id] = result
                
    def _calculate_strategy_metrics(self, signals: List[Dict], data: Dict) -> Dict[str, float]:
        """Calculate performance metrics for a strategy."""
        if not signals:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
            
        # Extract trade results
        returns = []
        wins = 0
        losses = 0
        
        for i, signal in enumerate(signals):
            if i > 0:  # Calculate return from previous signal
                if 'px' in signal and 'px' in signals[i-1]:
                    ret = (signal['px'] - signals[i-1]['px']) / signals[i-1]['px']
                    returns.append(ret)
                    if ret > 0:
                        wins += 1
                    elif ret < 0:
                        losses += 1
                        
        returns = np.array(returns) if returns else np.array([0])
        
        # Calculate metrics
        metrics = {
            'total_trades': len(signals),
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'avg_return': np.mean(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252 * 390) if np.std(returns) > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'profit_factor': abs(sum(r for r in returns if r > 0) / sum(r for r in returns if r < 0)) if any(r < 0 for r in returns) else 0
        }
        
        return metrics
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series."""
        if len(returns) == 0:
            return 0
            
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
        
    def _extract_regime_performance(self, signals: List[Dict], strategy_id: str) -> Dict[str, Dict[str, float]]:
        """Extract performance metrics by regime."""
        # This would need regime information correlated with signals
        # For now, return empty dict - would be populated with actual regime data
        return {}
        
    def _calculate_regime_stability(self, changes: List[Dict]) -> float:
        """Calculate how stable regime classifications are."""
        if len(changes) < 2:
            return 1.0
            
        # Calculate average duration between changes
        durations = []
        for i in range(1, len(changes)):
            duration = changes[i]['idx'] - changes[i-1]['idx']
            durations.append(duration)
            
        avg_duration = np.mean(durations) if durations else 1
        # Higher average duration = more stable
        stability = min(avg_duration / 10, 1.0)  # Normalize to 0-1
        
        return stability
        
    def _calculate_regime_durations(self, changes: List[Dict]) -> Dict[str, float]:
        """Calculate average duration for each regime."""
        regime_durations = {}
        current_regime = None
        start_idx = 0
        
        for change in changes:
            regime = change.get('val', 'unknown')
            idx = change.get('idx', 0)
            
            if current_regime is not None:
                duration = idx - start_idx
                if current_regime not in regime_durations:
                    regime_durations[current_regime] = []
                regime_durations[current_regime].append(duration)
                
            current_regime = regime
            start_idx = idx
            
        # Calculate averages
        avg_durations = {}
        for regime, durations in regime_durations.items():
            avg_durations[regime] = np.mean(durations) if durations else 0
            
        return avg_durations
        
    def analyze_strategy_performance(self) -> pd.DataFrame:
        """Analyze and rank strategy configurations by performance."""
        results = []
        
        for strat_id, result in self.strategy_results.items():
            row = {
                'strategy_id': strat_id,
                'strategy_type': result.strategy_type,
                **result.parameters,
                **result.metrics
            }
            results.append(row)
            
        df = pd.DataFrame(results)
        
        # Calculate composite score
        if len(df) > 0:
            df['composite_score'] = (
                df['sharpe_ratio'] * 0.3 +
                df['win_rate'] * 0.3 +
                (1 + df['max_drawdown']) * 0.2 +  # Less negative is better
                df['profit_factor'] * 0.2
            )
            
            # Rank strategies
            df = df.sort_values('composite_score', ascending=False)
            
        return df
        
    def analyze_classifier_stability(self) -> pd.DataFrame:
        """Analyze classifier configurations by stability and regime distribution."""
        results = []
        
        for class_id, result in self.classifier_results.items():
            row = {
                'classifier_id': class_id,
                'classifier_type': result.classifier_type,
                **result.parameters,
                'stability_score': result.stability_score,
                'num_regimes': len(result.regime_counts),
                **{f'regime_{k}_pct': v for k, v in result.regime_counts.items()}
            }
            results.append(row)
            
        df = pd.DataFrame(results)
        
        # Sort by stability
        if len(df) > 0:
            df = df.sort_values('stability_score', ascending=False)
            
        return df
        
    def find_optimal_pairings(self) -> pd.DataFrame:
        """Find optimal strategy-classifier pairings based on correlation analysis."""
        pairings = []
        
        # For each strategy, find best performing classifier
        for strat_id, strat_result in self.strategy_results.items():
            for class_id, class_result in self.classifier_results.items():
                # Calculate compatibility score
                score = self._calculate_pairing_score(strat_result, class_result)
                
                pairings.append({
                    'strategy_id': strat_id,
                    'strategy_type': strat_result.strategy_type,
                    'classifier_id': class_id,
                    'classifier_type': class_result.classifier_type,
                    'pairing_score': score,
                    'strategy_sharpe': strat_result.metrics.get('sharpe_ratio', 0),
                    'classifier_stability': class_result.stability_score
                })
                
        df = pd.DataFrame(pairings)
        
        # Sort by pairing score
        if len(df) > 0:
            df = df.sort_values('pairing_score', ascending=False)
            
        return df
        
    def _calculate_pairing_score(self, strategy: StrategyResult, classifier: ClassifierResult) -> float:
        """Calculate compatibility score between strategy and classifier."""
        # Simple scoring based on strategy performance and classifier stability
        # In practice, this would use regime-specific performance data
        
        strat_score = strategy.metrics.get('sharpe_ratio', 0) * 0.5 + strategy.metrics.get('win_rate', 0) * 0.5
        class_score = classifier.stability_score
        
        # Combine scores
        pairing_score = strat_score * 0.7 + class_score * 0.3
        
        return pairing_score
        
    def create_regime_performance_heatmap(self) -> None:
        """Create heatmap showing strategy performance by regime."""
        # Collect regime performance data
        regime_data = {}
        
        for strat_id, result in self.strategy_results.items():
            if result.regime_performance:
                regime_data[strat_id] = result.regime_performance
                
        if not regime_data:
            print("No regime performance data available")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(regime_data).T
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
        plt.title('Strategy Performance by Market Regime')
        plt.xlabel('Market Regime')
        plt.ylabel('Strategy Configuration')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'regime_performance_heatmap.png')
        plt.close()
        
    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("GRID SEARCH ANALYSIS REPORT")
        print("="*80)
        
        # Strategy performance analysis
        print("\n1. TOP PERFORMING STRATEGIES")
        print("-"*80)
        strat_df = self.analyze_strategy_performance()
        if len(strat_df) > 0:
            print(strat_df.head(10)[['strategy_id', 'strategy_type', 'sharpe_ratio', 
                                     'win_rate', 'max_drawdown', 'composite_score']])
        
        # Classifier stability analysis
        print("\n2. MOST STABLE CLASSIFIERS")
        print("-"*80)
        class_df = self.analyze_classifier_stability()
        if len(class_df) > 0:
            print(class_df.head(10)[['classifier_id', 'classifier_type', 
                                    'stability_score', 'num_regimes']])
        
        # Optimal pairings
        print("\n3. OPTIMAL STRATEGY-CLASSIFIER PAIRINGS")
        print("-"*80)
        pairing_df = self.find_optimal_pairings()
        if len(pairing_df) > 0:
            print(pairing_df.head(10)[['strategy_type', 'classifier_type', 
                                      'pairing_score', 'strategy_sharpe', 'classifier_stability']])
        
        # Parameter insights
        print("\n4. PARAMETER INSIGHTS")
        print("-"*80)
        self._analyze_parameter_importance(strat_df)
        
        # Save detailed results
        self._save_detailed_results(strat_df, class_df, pairing_df)
        
    def _analyze_parameter_importance(self, strat_df: pd.DataFrame) -> None:
        """Analyze which parameters have the most impact on performance."""
        if len(strat_df) == 0:
            return
            
        # Group by strategy type
        for strat_type in strat_df['strategy_type'].unique():
            print(f"\n{strat_type.upper()} Parameter Analysis:")
            
            type_df = strat_df[strat_df['strategy_type'] == strat_type]
            
            # Find parameter columns
            param_cols = [col for col in type_df.columns 
                         if col not in ['strategy_id', 'strategy_type', 'composite_score',
                                       'sharpe_ratio', 'win_rate', 'max_drawdown', 'profit_factor',
                                       'total_trades', 'avg_return']]
            
            # Calculate correlation with composite score
            for param in param_cols:
                if type_df[param].dtype in ['int64', 'float64']:
                    corr = type_df[param].corr(type_df['composite_score'])
                    if abs(corr) > 0.3:  # Only show significant correlations
                        print(f"  {param}: correlation = {corr:.3f}")
                        
    def _save_detailed_results(self, strat_df: pd.DataFrame, class_df: pd.DataFrame, 
                              pairing_df: pd.DataFrame) -> None:
        """Save detailed analysis results to files."""
        # Save DataFrames
        strat_df.to_csv(self.results_dir / 'strategy_rankings.csv', index=False)
        class_df.to_csv(self.results_dir / 'classifier_rankings.csv', index=False)
        pairing_df.to_csv(self.results_dir / 'optimal_pairings.csv', index=False)
        
        # Save best configurations as JSON for easy loading
        best_configs = {
            'best_strategies': strat_df.head(5).to_dict('records'),
            'best_classifiers': class_df.head(5).to_dict('records'),
            'best_pairings': pairing_df.head(5).to_dict('records'),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'best_configurations.json', 'w') as f:
            json.dump(best_configs, f, indent=2)
            
        print(f"\nDetailed results saved to: {self.results_dir}")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze grid search results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing grid search results')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top configurations to show')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = GridSearchAnalyzer(args.results_dir)
    analyzer.load_results()
    analyzer.generate_report()
    analyzer.create_regime_performance_heatmap()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()