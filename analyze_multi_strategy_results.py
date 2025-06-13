#!/usr/bin/env python3
"""
Multi-Strategy Results Analysis

Analyzes results from running multiple strategies and classifiers
to find optimal combinations and regime-specific performance.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns


class MultiStrategyAnalyzer:
    """Analyze results from multi-strategy signal generation."""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.strategy_signals = {}
        self.classifier_regimes = {}
        self.performance_matrix = None
        
    def load_all_results(self) -> None:
        """Load all signal and classification results from workspace."""
        print(f"Loading results from: {self.workspace_dir}")
        
        # Load strategy signals
        strategy_files = list(self.workspace_dir.glob("**/signals_strategy_*.json"))
        print(f"Found {len(strategy_files)} strategy signal files")
        
        for filepath in strategy_files:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Extract strategy results
            metadata = data.get('metadata', {})
            strategy_params = metadata.get('strategy_parameters', {})
            
            for strat_id, params in strategy_params.items():
                if 'classifier' not in strat_id:
                    self.strategy_signals[strat_id] = {
                        'type': params.get('type'),
                        'params': params.get('params', {}),
                        'signals': data.get('changes', []),
                        'total_bars': metadata.get('total_bars', 0)
                    }
                    
        # Load classifier regimes
        classifier_files = list(self.workspace_dir.glob("**/signals_classifier_*.json"))
        print(f"Found {len(classifier_files)} classifier regime files")
        
        for filepath in classifier_files:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Extract classifier results
            metadata = data.get('metadata', {})
            classifier_params = metadata.get('strategy_parameters', {})
            
            for class_id, params in classifier_params.items():
                if 'classifier' in class_id:
                    regime_breakdown = metadata.get('signal_statistics', {}).get('regime_breakdown', {})
                    self.classifier_regimes[class_id] = {
                        'type': params.get('type'),
                        'params': params.get('params', {}),
                        'regimes': data.get('changes', []),
                        'regime_counts': regime_breakdown,
                        'total_changes': len(data.get('changes', []))
                    }
                    
        print(f"Loaded {len(self.strategy_signals)} strategies")
        print(f"Loaded {len(self.classifier_regimes)} classifiers")
        
    def calculate_strategy_metrics(self) -> pd.DataFrame:
        """Calculate performance metrics for each strategy."""
        results = []
        
        for strat_id, strat_data in self.strategy_signals.items():
            signals = strat_data['signals']
            
            # Basic metrics
            metrics = {
                'strategy_id': strat_id,
                'strategy_type': strat_data['type'],
                'total_signals': len(signals),
                'signal_rate': len(signals) / strat_data['total_bars'] if strat_data['total_bars'] > 0 else 0
            }
            
            # Add parameters
            for param, value in strat_data['params'].items():
                metrics[f'param_{param}'] = value
                
            # Signal analysis
            if signals:
                # Count signal types
                long_signals = sum(1 for s in signals if s.get('val') == 'long')
                short_signals = sum(1 for s in signals if s.get('val') == 'short')
                flat_signals = sum(1 for s in signals if s.get('val') == 'flat')
                
                metrics.update({
                    'long_signals': long_signals,
                    'short_signals': short_signals,
                    'flat_signals': flat_signals,
                    'long_ratio': long_signals / len(signals) if len(signals) > 0 else 0,
                    'short_ratio': short_signals / len(signals) if len(signals) > 0 else 0
                })
                
                # Calculate average hold time (bars between signals)
                if len(signals) > 1:
                    hold_times = []
                    for i in range(1, len(signals)):
                        hold_time = signals[i]['idx'] - signals[i-1]['idx']
                        hold_times.append(hold_time)
                    metrics['avg_hold_time'] = np.mean(hold_times)
                else:
                    metrics['avg_hold_time'] = 0
                    
            results.append(metrics)
            
        return pd.DataFrame(results)
        
    def calculate_classifier_metrics(self) -> pd.DataFrame:
        """Calculate stability metrics for each classifier."""
        results = []
        
        for class_id, class_data in self.classifier_regimes.items():
            regimes = class_data['regimes']
            
            # Basic metrics
            metrics = {
                'classifier_id': class_id,
                'classifier_type': class_data['type'],
                'total_changes': class_data['total_changes'],
                'num_regimes': len(class_data['regime_counts'])
            }
            
            # Add parameters
            for param, value in class_data['params'].items():
                metrics[f'param_{param}'] = value
                
            # Regime distribution
            for regime, count in class_data['regime_counts'].items():
                metrics[f'regime_{regime}_count'] = count
                
            # Calculate stability (average duration in each regime)
            if len(regimes) > 1:
                durations = []
                for i in range(1, len(regimes)):
                    duration = regimes[i]['idx'] - regimes[i-1]['idx']
                    durations.append(duration)
                metrics['avg_regime_duration'] = np.mean(durations)
                metrics['stability_score'] = min(np.mean(durations) / 10, 1.0)  # Normalize
            else:
                metrics['avg_regime_duration'] = float('inf')
                metrics['stability_score'] = 1.0
                
            results.append(metrics)
            
        return pd.DataFrame(results)
        
    def analyze_strategy_classifier_combinations(self) -> pd.DataFrame:
        """Analyze how strategies would perform under different classifier regimes."""
        combinations = []
        
        for strat_id, strat_data in self.strategy_signals.items():
            for class_id, class_data in self.classifier_regimes.items():
                # For each combination, analyze potential performance
                combo_metrics = {
                    'strategy_id': strat_id,
                    'strategy_type': strat_data['type'],
                    'classifier_id': class_id,
                    'classifier_type': class_data['type'],
                    'strategy_signals': len(strat_data['signals']),
                    'classifier_changes': class_data['total_changes'],
                    'classifier_stability': class_data.get('stability_score', 0)
                }
                
                # Add compatibility score based on signal patterns and regime stability
                # Higher score = better combination
                signal_rate = len(strat_data['signals']) / strat_data['total_bars'] if strat_data['total_bars'] > 0 else 0
                stability = class_data.get('stability_score', 0)
                
                # Strategies with moderate signal rates work better with stable classifiers
                if signal_rate > 0.01 and signal_rate < 0.1:  # 1-10% signal rate
                    signal_quality = 1.0
                elif signal_rate > 0.1:  # Too many signals
                    signal_quality = 0.5
                else:  # Too few signals
                    signal_quality = 0.3
                    
                combo_metrics['compatibility_score'] = signal_quality * 0.6 + stability * 0.4
                
                combinations.append(combo_metrics)
                
        return pd.DataFrame(combinations)
        
    def create_performance_heatmap(self, metric: str = 'compatibility_score') -> None:
        """Create heatmap of strategy-classifier performance."""
        print("Heatmap generation disabled (matplotlib not available)")
        
    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("MULTI-STRATEGY ANALYSIS REPORT")
        print("="*80)
        
        # Strategy analysis
        print("\n1. STRATEGY PERFORMANCE METRICS")
        print("-"*80)
        strat_df = self.calculate_strategy_metrics()
        if not strat_df.empty:
            strat_df = strat_df.sort_values('signal_rate', ascending=False)
            print(strat_df[['strategy_id', 'strategy_type', 'total_signals', 
                           'signal_rate', 'long_ratio', 'short_ratio', 'avg_hold_time']].head(10))
            
        # Classifier analysis
        print("\n2. CLASSIFIER STABILITY METRICS")
        print("-"*80)
        class_df = self.calculate_classifier_metrics()
        if not class_df.empty:
            class_df = class_df.sort_values('stability_score', ascending=False)
            print(class_df[['classifier_id', 'classifier_type', 'total_changes',
                           'avg_regime_duration', 'stability_score']].head(10))
            
        # Best combinations
        print("\n3. OPTIMAL STRATEGY-CLASSIFIER COMBINATIONS")
        print("-"*80)
        combo_df = self.analyze_strategy_classifier_combinations()
        if not combo_df.empty:
            combo_df = combo_df.sort_values('compatibility_score', ascending=False)
            print(combo_df[['strategy_type', 'classifier_type', 
                           'compatibility_score']].head(10))
            
        # Parameter insights
        print("\n4. PARAMETER INSIGHTS")
        print("-"*80)
        self._analyze_parameter_effects(strat_df, class_df)
        
        # Save detailed results
        output_dir = self.workspace_dir / 'analysis_results'
        output_dir.mkdir(exist_ok=True)
        
        strat_df.to_csv(output_dir / 'strategy_metrics.csv', index=False)
        class_df.to_csv(output_dir / 'classifier_metrics.csv', index=False)
        combo_df.to_csv(output_dir / 'optimal_combinations.csv', index=False)
        
        # Save best configurations
        best_configs = {
            'best_strategy': strat_df.iloc[0].to_dict() if not strat_df.empty else {},
            'most_stable_classifier': class_df.iloc[0].to_dict() if not class_df.empty else {},
            'best_combination': combo_df.iloc[0].to_dict() if not combo_df.empty else {},
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'best_configurations.json', 'w') as f:
            json.dump(best_configs, f, indent=2)
            
        print(f"\nDetailed results saved to: {output_dir}")
        
    def _analyze_parameter_effects(self, strat_df: pd.DataFrame, class_df: pd.DataFrame) -> None:
        """Analyze which parameters have the most impact."""
        
        # Strategy parameters
        if not strat_df.empty:
            print("\nStrategy Parameter Effects:")
            
            # Group by strategy type
            for strat_type in strat_df['strategy_type'].unique():
                type_df = strat_df[strat_df['strategy_type'] == strat_type]
                
                # Find parameter columns
                param_cols = [col for col in type_df.columns if col.startswith('param_')]
                
                if param_cols:
                    print(f"\n{strat_type}:")
                    for param in param_cols:
                        if type_df[param].dtype in ['int64', 'float64']:
                            # Correlate with signal rate
                            corr = type_df[param].corr(type_df['signal_rate'])
                            if abs(corr) > 0.3:
                                print(f"  {param}: correlation with signal_rate = {corr:.3f}")
                                
        # Classifier parameters
        if not class_df.empty:
            print("\nClassifier Parameter Effects:")
            
            for class_type in class_df['classifier_type'].unique():
                type_df = class_df[class_df['classifier_type'] == class_type]
                
                param_cols = [col for col in type_df.columns if col.startswith('param_')]
                
                if param_cols:
                    print(f"\n{class_type}:")
                    for param in param_cols:
                        if type_df[param].dtype in ['int64', 'float64']:
                            # Correlate with stability
                            corr = type_df[param].corr(type_df['stability_score'])
                            if abs(corr) > 0.3:
                                print(f"  {param}: correlation with stability = {corr:.3f}")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze multi-strategy results')
    parser.add_argument('--workspace', type=str, required=True,
                       help='Workspace directory containing results')
    parser.add_argument('--create-heatmap', action='store_true',
                       help='Create strategy-classifier heatmap')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = MultiStrategyAnalyzer(args.workspace)
    analyzer.load_all_results()
    analyzer.generate_report()
    
    if args.create_heatmap:
        analyzer.create_performance_heatmap()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()