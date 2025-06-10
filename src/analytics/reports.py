"""
Minimal Report Generator

Creates simple reports from container metrics without duplicating
the sophisticated reporting already in tmp/analytics/basic_report.py
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

from .metrics import MetricsResult
from .patterns import Pattern

logger = logging.getLogger(__name__)


class MinimalReportGenerator:
    """
    Generate simple analytics reports focused on immediate insights.
    
    This is intentionally minimal - for sophisticated reporting,
    use the existing BacktestReportGenerator in tmp/analytics/basic_report.py
    """
    
    def __init__(self, output_dir: str = "./analytics_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_summary_report(
        self, 
        metrics_results: List[MetricsResult], 
        patterns: List[Pattern] = None
    ) -> str:
        """
        Generate a simple text summary report.
        
        Args:
            metrics_results: Metrics extracted from containers
            patterns: Optional detected patterns
            
        Returns:
            Path to generated report file
        """
        if not metrics_results:
            return self._generate_empty_report()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'container_id': r.container_id,
                'correlation_id': r.correlation_id,
                'total_return': r.total_return,
                'sharpe_ratio': r.sharpe_ratio,
                'max_drawdown': r.max_drawdown,
                'trade_count': r.trade_count,
                'win_rate': r.win_rate,
                'events_observed': r.events_observed,
                'events_pruned': r.events_pruned,
                'retention_policy': r.retention_policy
            }
            for r in metrics_results
        ])
        
        # Generate report content
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ADMF-PC ANALYTICS SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Containers Analyzed: {len(metrics_results)}")
        report_lines.append("")
        
        # Overall statistics
        report_lines.extend(self._generate_overall_stats(df))
        
        # Top performers
        report_lines.extend(self._generate_top_performers(df))
        
        # Memory efficiency stats
        report_lines.extend(self._generate_memory_stats(df))
        
        # Pattern insights (if provided)
        if patterns:
            report_lines.extend(self._generate_pattern_summary(patterns))
        
        # Event tracing insights
        report_lines.extend(self._generate_tracing_insights(df))
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"analytics_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Generated analytics report: {report_path}")
        return str(report_path)
    
    def _generate_overall_stats(self, df: pd.DataFrame) -> List[str]:
        """Generate overall statistics section."""
        lines = []
        lines.append("OVERALL STATISTICS")
        lines.append("-" * 40)
        
        # Basic stats
        total_containers = len(df)
        profitable_containers = len(df[df['total_return'] > 0])
        avg_return = df['total_return'].mean()
        avg_sharpe = df['sharpe_ratio'].mean()
        avg_trades = df['trade_count'].mean()
        
        lines.append(f"Total Containers: {total_containers}")
        lines.append(f"Profitable Containers: {profitable_containers} ({profitable_containers/total_containers:.1%})")
        lines.append(f"Average Return: {avg_return:.2f}%")
        lines.append(f"Average Sharpe Ratio: {avg_sharpe:.3f}")
        lines.append(f"Average Trades per Container: {avg_trades:.1f}")
        lines.append("")
        
        return lines
    
    def _generate_top_performers(self, df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Generate top performers section."""
        lines = []
        lines.append("TOP PERFORMERS")
        lines.append("-" * 40)
        
        # Sort by total return
        top_performers = df.nlargest(top_n, 'total_return')
        
        lines.append(f"{'Rank':<5} {'Container ID':<20} {'Return%':<10} {'Sharpe':<8} {'Trades':<8} {'Correlation ID':<15}")
        lines.append("-" * 80)
        
        for i, (_, row) in enumerate(top_performers.iterrows(), 1):
            container_id = str(row['container_id'])[:19]  # Truncate long IDs
            correlation_id = str(row['correlation_id'] or 'N/A')[:14]
            
            lines.append(
                f"{i:<5} {container_id:<20} {row['total_return']:>8.2f}% "
                f"{row['sharpe_ratio']:>7.3f} {row['trade_count']:>7.0f} {correlation_id:<15}"
            )
        
        lines.append("")
        
        # Insights
        if len(top_performers) > 0:
            best_performer = top_performers.iloc[0]
            lines.append(f"Best Performance: {best_performer['total_return']:.2f}% return")
            if best_performer['correlation_id']:
                lines.append(f"Event Trace Available: correlation_id = {best_performer['correlation_id']}")
                lines.append("Use this correlation_id for deep-dive event analysis")
        
        lines.append("")
        return lines
    
    def _generate_memory_stats(self, df: pd.DataFrame) -> List[str]:
        """Generate memory efficiency statistics."""
        lines = []
        lines.append("MEMORY EFFICIENCY ANALYSIS")
        lines.append("-" * 40)
        
        total_events_observed = df['events_observed'].sum()
        total_events_pruned = df['events_pruned'].sum()
        
        if total_events_observed > 0:
            pruning_efficiency = (total_events_pruned / total_events_observed) * 100
            lines.append(f"Total Events Observed: {total_events_observed:,}")
            lines.append(f"Total Events Pruned: {total_events_pruned:,}")
            lines.append(f"Memory Pruning Efficiency: {pruning_efficiency:.1f}%")
        else:
            lines.append("No event statistics available")
        
        # Retention policy breakdown
        retention_counts = df['retention_policy'].value_counts()
        lines.append("")
        lines.append("Retention Policies Used:")
        for policy, count in retention_counts.items():
            lines.append(f"  {policy}: {count} containers")
        
        lines.append("")
        return lines
    
    def _generate_pattern_summary(self, patterns: List[Pattern]) -> List[str]:
        """Generate pattern analysis summary."""
        lines = []
        lines.append("PATTERN ANALYSIS")
        lines.append("-" * 40)
        
        if not patterns:
            lines.append("No significant patterns detected")
            lines.append("")
            return lines
        
        for i, pattern in enumerate(patterns, 1):
            lines.append(f"{i}. {pattern.description}")
            lines.append(f"   Success Rate: {pattern.success_rate:.1%}")
            lines.append(f"   Sample Size: {pattern.sample_count}")
            lines.append(f"   Avg Return: {pattern.avg_return:.2f}%")
            lines.append(f"   Event Traces: {len(pattern.correlation_ids)} correlation IDs available")
            lines.append("")
        
        # Best pattern recommendation
        if patterns:
            best_pattern = max(patterns, key=lambda p: p.success_rate * p.avg_return)
            lines.append("RECOMMENDATION:")
            lines.append(f"Focus on '{best_pattern.pattern_type}' pattern")
            lines.append(f"({best_pattern.success_rate:.1%} success rate, {best_pattern.avg_return:.2f}% avg return)")
            lines.append("")
        
        return lines
    
    def _generate_tracing_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate event tracing insights."""
        lines = []
        lines.append("EVENT TRACING INSIGHTS")
        lines.append("-" * 40)
        
        # Correlation ID availability
        has_correlation_id = df['correlation_id'].notna().sum()
        total_containers = len(df)
        
        lines.append(f"Containers with Correlation IDs: {has_correlation_id}/{total_containers}")
        
        if has_correlation_id > 0:
            lines.append("")
            lines.append("DEEP ANALYSIS READY:")
            lines.append("Use correlation IDs to analyze:")
            lines.append("  - Signal generation patterns")
            lines.append("  - Risk management decisions")
            lines.append("  - Execution quality")
            lines.append("  - Strategy behavior under different market conditions")
            lines.append("")
            lines.append("Next Steps:")
            lines.append("1. Export correlation IDs for promising containers")
            lines.append("2. Use event trace analysis tools")
            lines.append("3. Investigate common patterns in event sequences")
        else:
            lines.append("Enable event tracing for deeper analysis capabilities")
        
        lines.append("")
        return lines
    
    def _generate_empty_report(self) -> str:
        """Generate report when no metrics are available."""
        lines = [
            "=" * 80,
            "ADMF-PC ANALYTICS SUMMARY REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "No metrics data available for analysis.",
            "",
            "Possible reasons:",
            "- No containers were executed",
            "- Containers haven't completed execution",
            "- Event observers not properly configured",
            "",
            "Recommendations:",
            "1. Check container execution status",
            "2. Verify event observer setup",
            "3. Ensure metrics calculation is enabled",
            ""
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"analytics_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return str(report_path)
    
    def export_correlation_ids(self, metrics_results: List[MetricsResult], criteria: Dict[str, Any] = None) -> str:
        """
        Export correlation IDs for promising containers.
        
        Args:
            metrics_results: Metrics from containers
            criteria: Optional filtering criteria
            
        Returns:
            Path to exported correlation IDs file
        """
        # Apply filtering criteria
        filtered_results = metrics_results
        
        if criteria:
            df = pd.DataFrame([
                {
                    'total_return': r.total_return,
                    'sharpe_ratio': r.sharpe_ratio,
                    'correlation_id': r.correlation_id,
                    'result': r
                }
                for r in metrics_results if r.correlation_id
            ])
            
            # Apply filters
            if 'min_return' in criteria:
                df = df[df['total_return'] >= criteria['min_return']]
            if 'min_sharpe' in criteria:
                df = df[df['sharpe_ratio'] >= criteria['min_sharpe']]
            
            filtered_results = df['result'].tolist()
        
        # Export correlation IDs
        correlation_ids = [r.correlation_id for r in filtered_results if r.correlation_id]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.output_dir / f"correlation_ids_{timestamp}.txt"
        
        with open(export_path, 'w') as f:
            f.write("# Correlation IDs for Deep Analysis\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Criteria: {criteria or 'None'}\n")
            f.write(f"# Count: {len(correlation_ids)}\n\n")
            
            for cid in correlation_ids:
                f.write(f"{cid}\n")
        
        self.logger.info(f"Exported {len(correlation_ids)} correlation IDs to {export_path}")
        return str(export_path)