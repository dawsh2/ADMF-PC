"""
Strategy analysis module for sparse trace data.

Combines performance calculation, regime attribution, and classifier analysis
to provide comprehensive strategy performance analysis by market regime.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json

from .performance_calculation import (
    calculate_log_returns_with_costs,
    ExecutionCostConfig,
    summarize_performance
)
from .regime_attribution import (
    RegimeAttributor,
    attribute_trades_to_regimes,
    group_trades_by_regime,
    validate_regime_attribution
)
from .classifier_analysis import ClassifierAnalyzer


def load_strategy_signals(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Load strategy signal data from parquet file.
    
    Args:
        file_path: Path to strategy parquet file
        
    Returns:
        DataFrame with normalized columns or None if error
    """
    try:
        df = pd.read_parquet(file_path)
        
        # Normalize column names
        df = df.rename(columns={
            'idx': 'bar_idx',
            'val': 'signal_value',
            'px': 'price'
        })
        
        # Ensure required columns exist
        required_cols = ['bar_idx', 'signal_value', 'price']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns in {file_path}")
            return None
        
        return df[required_cols].sort_values('bar_idx')
        
    except Exception as e:
        print(f"Error loading strategy file {file_path}: {e}")
        return None


def analyze_strategy_performance_by_regime(
    strategy_file: Path,
    classifier_name: str,
    workspace_path: Path,
    cost_config: Optional[ExecutionCostConfig] = None,
    initial_capital: float = 10000.0
) -> Optional[Dict[str, Any]]:
    """
    Analyze strategy performance broken down by market regime.
    
    Args:
        strategy_file: Path to strategy signal file
        classifier_name: Name of classifier for regime attribution
        workspace_path: Path to workspace containing trace data
        cost_config: Optional execution cost configuration
        initial_capital: Starting capital for calculations
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    # Load strategy signals
    signals_df = load_strategy_signals(strategy_file)
    if signals_df is None:
        return None
    
    # Calculate overall performance
    overall_performance = calculate_log_returns_with_costs(
        signals_df, cost_config, initial_capital
    )
    
    if overall_performance['num_trades'] == 0:
        return {
            'strategy_file': str(strategy_file),
            'strategy_name': strategy_file.stem,
            'classifier_name': classifier_name,
            'overall_performance': overall_performance,
            'regime_performance': {},
            'regime_attribution': {
                'total_trades': 0,
                'trades_with_regime': 0,
                'attribution_accuracy': 1.0
            }
        }
    
    # Attribute trades to regimes
    attributor = RegimeAttributor(workspace_path)
    attributed_trades = attributor.attribute_strategy_trades(
        overall_performance['trades'],
        classifier_name
    )
    
    if attributed_trades is None:
        print(f"Could not attribute trades to regimes using classifier: {classifier_name}")
        return None
    
    # Validate attribution
    classifier_changes = attributor.load_classifier_changes(classifier_name)
    if classifier_changes is not None:
        attribution_validation = validate_regime_attribution(attributed_trades, classifier_changes)
    else:
        attribution_validation = {'attribution_accuracy': 0.0}
    
    # Group trades by regime and calculate regime-specific performance
    regime_trades = group_trades_by_regime(attributed_trades)
    regime_performance = {}
    
    for regime, trades in regime_trades.items():
        if not trades:
            continue
        
        # Calculate performance metrics for this regime
        net_returns = [trade.get('net_log_return', 0) for trade in trades]
        gross_returns = [trade.get('gross_log_return', 0) for trade in trades]
        
        total_net_return = sum(net_returns)
        total_gross_return = sum(gross_returns)
        
        # Convert to percentage returns
        net_percentage = np.exp(total_net_return) - 1 if total_net_return > -10 else -0.99
        gross_percentage = np.exp(total_gross_return) - 1 if total_gross_return > -10 else -0.99
        
        # Calculate statistics
        winning_trades = [r for r in net_returns if r > 0]
        losing_trades = [r for r in net_returns if r < 0]
        
        regime_performance[regime] = {
            'trade_count': len(trades),
            'total_net_log_return': total_net_return,
            'total_gross_log_return': total_gross_return,
            'net_percentage_return': net_percentage,
            'gross_percentage_return': gross_percentage,
            'avg_trade_return': total_net_return / len(trades),
            'win_rate': len(winning_trades) / len(trades),
            'avg_winning_trade': np.mean(winning_trades) if winning_trades else 0,
            'avg_losing_trade': np.mean(losing_trades) if losing_trades else 0,
            'best_trade': max(net_returns),
            'worst_trade': min(net_returns),
            'total_bars_held': sum(trade.get('bars_held', 0) for trade in trades),
            'avg_bars_held': np.mean([trade.get('bars_held', 0) for trade in trades]),
            'profit_factor': (
                abs(sum(winning_trades)) / abs(sum(losing_trades))
                if losing_trades else float('inf')
            )
        }
    
    return {
        'strategy_file': str(strategy_file),
        'strategy_name': strategy_file.stem,
        'classifier_name': classifier_name,
        'overall_performance': overall_performance,
        'regime_performance': regime_performance,
        'regime_attribution': attribution_validation,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }


class StrategyAnalyzer:
    """Main class for comprehensive strategy analysis by regime."""
    
    def __init__(self, workspace_path: Path):
        """
        Initialize strategy analyzer.
        
        Args:
            workspace_path: Path to workspace containing trace data
        """
        self.workspace_path = Path(workspace_path)
        self.signals_dir = self.workspace_path / "traces" / "SPY_1m" / "signals"
        self.attributor = RegimeAttributor(workspace_path)
        self.classifier_analyzer = ClassifierAnalyzer(workspace_path)
        
    def find_strategy_files(
        self,
        strategy_pattern: Optional[str] = None,
        strategy_types: Optional[List[str]] = None,
        max_files: Optional[int] = None
    ) -> List[Path]:
        """
        Find strategy files matching criteria.
        
        Args:
            strategy_pattern: Pattern to match in strategy names (e.g., "macd")
            strategy_types: List of strategy types to include
            max_files: Maximum number of files to return
            
        Returns:
            List of strategy file paths
        """
        if not self.signals_dir.exists():
            print(f"Signals directory not found: {self.signals_dir}")
            return []
        
        strategy_files = list(self.signals_dir.rglob("*.parquet"))
        
        # Filter by pattern
        if strategy_pattern:
            strategy_files = [
                f for f in strategy_files 
                if strategy_pattern.lower() in f.stem.lower()
            ]
        
        # Filter by strategy types
        if strategy_types:
            filtered_files = []
            for file_path in strategy_files:
                for strategy_type in strategy_types:
                    if strategy_type in str(file_path.parent):
                        filtered_files.append(file_path)
                        break
            strategy_files = filtered_files
        
        # Limit number of files
        if max_files:
            strategy_files = strategy_files[:max_files]
        
        return strategy_files
    
    def analyze_multiple_strategies(
        self,
        strategy_files: List[Path],
        classifier_name: str,
        cost_config: Optional[ExecutionCostConfig] = None,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze multiple strategies by regime.
        
        Args:
            strategy_files: List of strategy file paths to analyze
            classifier_name: Name of classifier for regime attribution
            cost_config: Optional execution cost configuration
            parallel: Whether to use parallel processing (not implemented yet)
            
        Returns:
            Dictionary with analysis results for all strategies
        """
        results = {
            'classifier_name': classifier_name,
            'num_strategies': len(strategy_files),
            'cost_config': cost_config.__dict__ if cost_config else None,
            'strategies': {},
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        print(f"Analyzing {len(strategy_files)} strategies using classifier: {classifier_name}")
        
        for i, strategy_file in enumerate(strategy_files):
            print(f"  Progress: {i+1}/{len(strategy_files)} - {strategy_file.stem}")
            
            strategy_analysis = analyze_strategy_performance_by_regime(
                strategy_file, classifier_name, self.workspace_path, cost_config
            )
            
            if strategy_analysis is not None:
                results['strategies'][strategy_file.stem] = strategy_analysis
            else:
                print(f"    Warning: Could not analyze {strategy_file.stem}")
        
        return results
    
    def compare_strategies_by_regime(
        self,
        analysis_results: Dict[str, Any],
        regime_name: str,
        sort_by: str = 'net_percentage_return',
        top_n: int = 10
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Compare strategies within a specific regime.
        
        Args:
            analysis_results: Results from analyze_multiple_strategies
            regime_name: Name of regime to compare
            sort_by: Metric to sort by
            top_n: Number of top performers to return
            
        Returns:
            List of (strategy_name, regime_performance) tuples
        """
        regime_results = []
        
        for strategy_name, strategy_data in analysis_results['strategies'].items():
            regime_perf = strategy_data.get('regime_performance', {}).get(regime_name)
            
            if regime_perf and regime_perf['trade_count'] > 0:
                regime_results.append((strategy_name, regime_perf))
        
        # Sort by specified metric
        if sort_by in ['net_percentage_return', 'total_net_log_return', 'avg_trade_return']:
            regime_results.sort(key=lambda x: x[1][sort_by], reverse=True)
        elif sort_by == 'win_rate':
            regime_results.sort(key=lambda x: x[1][sort_by], reverse=True)
        
        return regime_results[:top_n]
    
    def print_strategy_comparison(
        self,
        analysis_results: Dict[str, Any],
        regime_name: str,
        sort_by: str = 'net_percentage_return',
        top_n: int = 10
    ) -> None:
        """
        Print formatted comparison of strategies within a regime.
        
        Args:
            analysis_results: Results from analyze_multiple_strategies
            regime_name: Name of regime to compare
            sort_by: Metric to sort by
            top_n: Number of top performers to show
        """
        top_performers = self.compare_strategies_by_regime(
            analysis_results, regime_name, sort_by, top_n
        )
        
        if not top_performers:
            print(f"No strategies found with trades in {regime_name} regime")
            return
        
        print(f"\n{'='*90}")
        print(f"TOP {top_n} STRATEGIES IN {regime_name.upper()} REGIME")
        print(f"Sorted by: {sort_by}")
        print("="*90)
        
        print(f"{'Rank':<5} {'Strategy':<40} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Avg Hold':<10}")
        print("-" * 90)
        
        for i, (strategy_name, regime_perf) in enumerate(top_performers):
            rank = i + 1
            return_val = regime_perf[sort_by]
            trades = regime_perf['trade_count']
            win_rate = regime_perf['win_rate']
            avg_hold = regime_perf['avg_bars_held']
            
            if sort_by in ['net_percentage_return', 'gross_percentage_return']:
                return_str = f"{return_val:.2%}"
            else:
                return_str = f"{return_val:.4f}"
            
            print(f"{rank:<5} {strategy_name:<40} {return_str:<10} {trades:<8} "
                  f"{win_rate:.2%}   {avg_hold:<10.1f}")
    
    def generate_regime_summary_report(
        self,
        analysis_results: Dict[str, Any],
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generate comprehensive summary report of regime analysis.
        
        Args:
            analysis_results: Results from analyze_multiple_strategies
            output_file: Optional file to save report
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "="*90,
            "STRATEGY PERFORMANCE BY REGIME - SUMMARY REPORT",
            "="*90,
            f"Analysis Date: {analysis_results['analysis_timestamp']}",
            f"Classifier: {analysis_results['classifier_name']}",
            f"Strategies Analyzed: {analysis_results['num_strategies']}",
            ""
        ]
        
        # Collect regime statistics across all strategies
        regime_stats = {}
        
        for strategy_name, strategy_data in analysis_results['strategies'].items():
            for regime, regime_perf in strategy_data.get('regime_performance', {}).items():
                if regime not in regime_stats:
                    regime_stats[regime] = {
                        'strategy_count': 0,
                        'total_trades': 0,
                        'total_return': 0,
                        'returns': []
                    }
                
                regime_stats[regime]['strategy_count'] += 1
                regime_stats[regime]['total_trades'] += regime_perf['trade_count']
                regime_stats[regime]['total_return'] += regime_perf['net_percentage_return']
                regime_stats[regime]['returns'].append(regime_perf['net_percentage_return'])
        
        # Generate regime summaries
        for regime, stats in regime_stats.items():
            if stats['strategy_count'] == 0:
                continue
                
            avg_return = stats['total_return'] / stats['strategy_count']
            median_return = np.median(stats['returns'])
            
            report_lines.extend([
                f"{regime.upper()} REGIME:",
                f"  Strategies with trades: {stats['strategy_count']}",
                f"  Total trades: {stats['total_trades']}",
                f"  Average return per strategy: {avg_return:.2%}",
                f"  Median return per strategy: {median_return:.2%}",
                ""
            ])
        
        # Generate top performers section
        for regime in regime_stats.keys():
            top_performers = self.compare_strategies_by_regime(
                analysis_results, regime, 'net_percentage_return', 5
            )
            
            if top_performers:
                report_lines.extend([
                    f"TOP 5 PERFORMERS - {regime.upper()} REGIME:",
                    f"{'Rank':<5} {'Strategy':<40} {'Return':<10} {'Trades':<8}",
                    "-" * 70
                ])
                
                for i, (strategy_name, regime_perf) in enumerate(top_performers):
                    return_val = regime_perf['net_percentage_return']
                    trades = regime_perf['trade_count']
                    
                    report_lines.append(
                        f"{i+1:<5} {strategy_name:<40} {return_val:.2%}   {trades:<8}"
                    )
                
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def save_analysis_results(
        self,
        analysis_results: Dict[str, Any],
        output_file: Path
    ) -> None:
        """
        Save analysis results to JSON file.
        
        Args:
            analysis_results: Results from analyze_multiple_strategies
            output_file: Path to output JSON file
        """
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"Analysis results saved to: {output_file}")


def load_strategy_analysis_results(file_path: Path) -> Dict[str, Any]:
    """
    Load previously saved strategy analysis results.
    
    Args:
        file_path: Path to JSON file with analysis results
        
    Returns:
        Dictionary with analysis results
    """
    with open(file_path, 'r') as f:
        return json.load(f)