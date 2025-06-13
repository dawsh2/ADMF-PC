"""
Grid Search Results Analyzer

Comprehensive framework for analyzing multi-dimensional grid search results
from strategy and classifier parameter expansions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GridSearchAnalyzer:
    """Analyze results from expansive grid search runs."""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.strategy_results = {}
        self.classifier_results = {}
        self.performance_metrics = {}
        self.regime_correlations = {}
        
    def load_all_results(self) -> Tuple[int, int]:
        """Load all signal and classification results from workspace."""
        strategy_files = list(self.workspace_dir.glob("signals_strategy_*.json"))
        classifier_files = list(self.workspace_dir.glob("signals_classifier_*.json"))
        
        logger.info(f"Loading {len(strategy_files)} strategy files and {len(classifier_files)} classifier files")
        
        # Load strategy results
        for filepath in strategy_files:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            strategy_params = metadata.get('strategy_parameters', {})
            
            for strat_id, params in strategy_params.items():
                if 'classifier' not in strat_id:
                    self.strategy_results[strat_id] = {
                        'type': params.get('type'),
                        'params': params.get('params', {}),
                        'signals': data.get('changes', []),
                        'total_bars': metadata.get('total_bars', 0),
                        'signal_stats': metadata.get('signal_statistics', {})
                    }
        
        # Load classifier results  
        for filepath in classifier_files:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            metadata = data.get('metadata', {})
            classifier_params = metadata.get('strategy_parameters', {})
            
            for class_id, params in classifier_params.items():
                if 'classifier' in class_id:
                    self.classifier_results[class_id] = {
                        'type': params.get('type'),
                        'params': params.get('params', {}),
                        'regimes': data.get('changes', []),
                        'regime_stats': metadata.get('signal_statistics', {})
                    }
                    
        return len(self.strategy_results), len(self.classifier_results)
    
    def calculate_strategy_performance(self) -> pd.DataFrame:
        """Calculate performance metrics for each strategy."""
        results = []
        
        for strat_id, strat_data in self.strategy_results.items():
            signals = strat_data['signals']
            total_bars = strat_data['total_bars']
            
            # Base metrics
            metrics = {
                'strategy_id': strat_id,
                'strategy_type': strat_data['type'],
                'total_signals': len(signals),
                'signal_rate': len(signals) / total_bars if total_bars > 0 else 0,
                'total_bars': total_bars
            }
            
            # Add parameters
            for param, value in strat_data['params'].items():
                metrics[f'param_{param}'] = value
            
            # Signal distribution
            if signals:
                signal_directions = defaultdict(int)
                holding_times = []
                signal_strengths = []
                
                for i, signal in enumerate(signals):
                    direction = signal.get('val', 'flat')
                    signal_directions[direction] += 1
                    signal_strengths.append(signal.get('strength', 0))
                    
                    # Calculate holding time
                    if i < len(signals) - 1:
                        hold_time = signals[i+1]['idx'] - signal['idx']
                        holding_times.append(hold_time)
                
                # Add signal metrics
                metrics.update({
                    'long_signals': signal_directions.get('long', 0),
                    'short_signals': signal_directions.get('short', 0),
                    'flat_signals': signal_directions.get('flat', 0),
                    'avg_signal_strength': np.mean(signal_strengths) if signal_strengths else 0,
                    'avg_holding_time': np.mean(holding_times) if holding_times else 0,
                    'signal_consistency': 1 - np.std(holding_times) / np.mean(holding_times) if holding_times and np.mean(holding_times) > 0 else 0
                })
                
                # Calculate signal quality score
                long_short_ratio = (signal_directions['long'] + signal_directions['short']) / len(signals) if len(signals) > 0 else 0
                metrics['signal_quality_score'] = (
                    long_short_ratio * 0.4 +  # Prefer strategies that take positions
                    metrics['signal_consistency'] * 0.3 +  # Consistent holding times
                    metrics['avg_signal_strength'] * 0.3  # Strong conviction signals
                )
            else:
                metrics.update({
                    'long_signals': 0,
                    'short_signals': 0,
                    'flat_signals': 0,
                    'avg_signal_strength': 0,
                    'avg_holding_time': 0,
                    'signal_consistency': 0,
                    'signal_quality_score': 0
                })
                
            results.append(metrics)
            
        return pd.DataFrame(results)
    
    def calculate_classifier_stability(self) -> pd.DataFrame:
        """Calculate stability metrics for each classifier."""
        results = []
        
        for class_id, class_data in self.classifier_results.items():
            regimes = class_data['regimes']
            
            metrics = {
                'classifier_id': class_id,
                'classifier_type': class_data['type'],
                'total_changes': len(regimes),
                'unique_regimes': len(set(r['val'] for r in regimes)) if regimes else 0
            }
            
            # Add parameters
            for param, value in class_data['params'].items():
                metrics[f'param_{param}'] = value
            
            # Calculate regime durations
            if len(regimes) > 1:
                durations = []
                regime_counts = defaultdict(int)
                
                for i in range(len(regimes) - 1):
                    duration = regimes[i+1]['idx'] - regimes[i]['idx']
                    durations.append(duration)
                    regime_counts[regimes[i]['val']] += 1
                
                # Add final regime
                regime_counts[regimes[-1]['val']] += 1
                
                metrics.update({
                    'avg_regime_duration': np.mean(durations),
                    'min_regime_duration': np.min(durations),
                    'max_regime_duration': np.max(durations),
                    'regime_stability': min(np.mean(durations) / 50, 1.0),  # Normalize by 50 bars
                    'regime_distribution': dict(regime_counts)
                })
            else:
                metrics.update({
                    'avg_regime_duration': float('inf'),
                    'min_regime_duration': float('inf'),
                    'max_regime_duration': float('inf'),
                    'regime_stability': 1.0,
                    'regime_distribution': {}
                })
                
            results.append(metrics)
            
        return pd.DataFrame(results)
    
    def analyze_parameter_sensitivity(self, strategy_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which parameters have the most impact on performance."""
        sensitivity_results = {}
        
        # Group by strategy type
        for strategy_type in strategy_df['strategy_type'].unique():
            type_df = strategy_df[strategy_df['strategy_type'] == strategy_type]
            
            # Find parameter columns
            param_cols = [col for col in type_df.columns if col.startswith('param_')]
            
            sensitivity_results[strategy_type] = {}
            
            for param in param_cols:
                if type_df[param].dtype in ['int64', 'float64']:
                    # Calculate correlation with performance metrics
                    correlations = {
                        'signal_rate': type_df[param].corr(type_df['signal_rate']),
                        'signal_quality_score': type_df[param].corr(type_df['signal_quality_score']),
                        'avg_holding_time': type_df[param].corr(type_df['avg_holding_time'])
                    }
                    
                    # Calculate variance explained
                    param_values = type_df[param].unique()
                    if len(param_values) > 1:
                        # Group by parameter value and calculate mean performance
                        grouped = type_df.groupby(param)['signal_quality_score'].agg(['mean', 'std'])
                        variance_ratio = grouped['std'].mean() / grouped['mean'].std() if grouped['mean'].std() > 0 else 0
                        
                        sensitivity_results[strategy_type][param] = {
                            'correlations': correlations,
                            'variance_ratio': variance_ratio,
                            'impact_score': abs(correlations['signal_quality_score']) * (1 - variance_ratio)
                        }
                        
        return sensitivity_results
    
    def find_optimal_configurations(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """Find the best performing strategy configurations."""
        strategy_df = self.calculate_strategy_performance()
        classifier_df = self.calculate_classifier_stability()
        
        # Sort strategies by quality score
        best_strategies = strategy_df.nlargest(top_n, 'signal_quality_score')
        
        # Sort classifiers by stability
        if not classifier_df.empty and 'regime_stability' in classifier_df.columns:
            best_classifiers = classifier_df.nlargest(top_n, 'regime_stability')
        else:
            best_classifiers = pd.DataFrame()
        
        # Find best combinations for each strategy type
        best_by_type = {}
        for strategy_type in strategy_df['strategy_type'].unique():
            type_df = strategy_df[strategy_df['strategy_type'] == strategy_type]
            best_by_type[strategy_type] = type_df.nlargest(3, 'signal_quality_score')
        
        return {
            'best_overall': best_strategies,
            'best_classifiers': best_classifiers,
            'best_by_type': best_by_type
        }
    
    def export_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """Export analysis results in multiple formats."""
        if output_dir is None:
            output_dir = self.workspace_dir / 'analysis'
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True)
        
        # Calculate all metrics
        strategy_df = self.calculate_strategy_performance()
        classifier_df = self.calculate_classifier_stability()
        optimal_configs = self.find_optimal_configurations()
        sensitivity = self.analyze_parameter_sensitivity(strategy_df)
        
        # Export DataFrames to CSV
        strategy_df.to_csv(output_dir / 'strategy_performance.csv', index=False)
        classifier_df.to_csv(output_dir / 'classifier_stability.csv', index=False)
        
        # Export optimal configurations
        for key, df in optimal_configs['best_by_type'].items():
            df.to_csv(output_dir / f'best_{key}_configs.csv', index=False)
        
        # Export summary JSON
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_strategies': len(self.strategy_results),
            'total_classifiers': len(self.classifier_results),
            'best_strategy': optimal_configs['best_overall'].iloc[0].to_dict() if not optimal_configs['best_overall'].empty else {},
            'best_classifier': optimal_configs['best_classifiers'].iloc[0].to_dict() if not optimal_configs['best_classifiers'].empty else {},
            'parameter_sensitivity': sensitivity,
            'workspace': str(self.workspace_dir)
        }
        
        with open(output_dir / 'grid_search_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate report
        report_path = output_dir / 'grid_search_report.txt'
        self._generate_report(report_path, strategy_df, classifier_df, optimal_configs, sensitivity)
        
        return {
            'strategy_performance': str(output_dir / 'strategy_performance.csv'),
            'classifier_stability': str(output_dir / 'classifier_stability.csv'),
            'summary': str(output_dir / 'grid_search_summary.json'),
            'report': str(report_path)
        }
    
    def _generate_report(self, report_path: Path, strategy_df: pd.DataFrame, 
                        classifier_df: pd.DataFrame, optimal_configs: Dict[str, Any],
                        sensitivity: Dict[str, Any]) -> None:
        """Generate human-readable report."""
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRID SEARCH ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Workspace: {self.workspace_dir}\n")
            f.write(f"Total Strategies Tested: {len(self.strategy_results)}\n")
            f.write(f"Total Classifiers Tested: {len(self.classifier_results)}\n\n")
            
            f.write("1. TOP PERFORMING STRATEGIES\n")
            f.write("-"*80 + "\n")
            best_strats = optimal_configs['best_overall'].head(5)
            for _, row in best_strats.iterrows():
                f.write(f"\nStrategy: {row['strategy_id']}\n")
                f.write(f"  Type: {row['strategy_type']}\n")
                f.write(f"  Quality Score: {row['signal_quality_score']:.3f}\n")
                f.write(f"  Signal Rate: {row['signal_rate']:.4f}\n")
                f.write(f"  Avg Holding Time: {row['avg_holding_time']:.1f} bars\n")
                
                # Write parameters
                params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_')}
                f.write(f"  Parameters: {params}\n")
            
            f.write("\n2. MOST STABLE CLASSIFIERS\n")
            f.write("-"*80 + "\n")
            best_class = optimal_configs['best_classifiers'].head(3)
            for _, row in best_class.iterrows():
                f.write(f"\nClassifier: {row['classifier_id']}\n")
                f.write(f"  Type: {row['classifier_type']}\n")
                f.write(f"  Stability Score: {row['regime_stability']:.3f}\n")
                f.write(f"  Avg Regime Duration: {row['avg_regime_duration']:.1f} bars\n")
                f.write(f"  Total Changes: {row['total_changes']}\n")
            
            f.write("\n3. PARAMETER SENSITIVITY ANALYSIS\n")
            f.write("-"*80 + "\n")
            for strategy_type, params in sensitivity.items():
                f.write(f"\n{strategy_type}:\n")
                sorted_params = sorted(params.items(), key=lambda x: x[1].get('impact_score', 0), reverse=True)
                for param, analysis in sorted_params[:3]:
                    f.write(f"  {param}: Impact Score = {analysis.get('impact_score', 0):.3f}\n")
                    f.write(f"    Quality Correlation: {analysis['correlations']['signal_quality_score']:.3f}\n")
            
            f.write("\n4. RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            f.write("Based on the grid search results:\n\n")
            
            # Best overall strategy
            if not optimal_configs['best_overall'].empty:
                best = optimal_configs['best_overall'].iloc[0]
                f.write(f"- Best Overall Strategy: {best['strategy_type']}\n")
                params = {k.replace('param_', ''): v for k, v in best.items() if k.startswith('param_')}
                f.write(f"  Optimal Parameters: {params}\n\n")
            
            # Most impactful parameters
            f.write("- Most Impactful Parameters:\n")
            for strategy_type in sensitivity:
                if sensitivity[strategy_type]:
                    most_impactful = max(sensitivity[strategy_type].items(), 
                                       key=lambda x: x[1].get('impact_score', 0))
                    f.write(f"  {strategy_type}: {most_impactful[0]}\n")


def main():
    """Run grid search analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze grid search results')
    parser.add_argument('--workspace', type=str, required=True,
                       help='Workspace directory containing results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for analysis results')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top configurations to show')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = GridSearchAnalyzer(args.workspace)
    n_strategies, n_classifiers = analyzer.load_all_results()
    
    print(f"Loaded {n_strategies} strategies and {n_classifiers} classifiers")
    
    # Export results
    output_files = analyzer.export_results(args.output)
    
    print("\nAnalysis complete! Results saved to:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")
    
    # Show top results
    optimal = analyzer.find_optimal_configurations(args.top_n)
    print(f"\nTop {args.top_n} Strategies:")
    print(optimal['best_overall'][['strategy_id', 'strategy_type', 'signal_quality_score']].head())


if __name__ == "__main__":
    main()