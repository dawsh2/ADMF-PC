"""
Simple Pattern Detector

SQL-first pattern discovery for optimization results.
Focuses on immediate actionable insights rather than complex mining.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Simple pattern representation."""
    pattern_type: str
    description: str
    success_rate: float
    sample_count: int
    avg_return: float
    conditions: Dict[str, Any]
    correlation_ids: List[str]  # For linking to event traces


class SimplePatternDetector:
    """
    Detect simple patterns from optimization results using SQL-like operations.
    
    Focuses on immediately actionable patterns rather than complex mining.
    Designed to work with the correlation ID bridge for deeper analysis.
    """
    
    def __init__(self, min_sample_size: int = 5, min_success_rate: float = 0.6):
        self.min_sample_size = min_sample_size
        self.min_success_rate = min_success_rate
        self.logger = logging.getLogger(__name__)
    
    def detect_patterns(self, metrics_df: pd.DataFrame) -> List[Pattern]:
        """
        Detect patterns from metrics DataFrame.
        
        Args:
            metrics_df: DataFrame with columns: container_id, correlation_id, 
                       total_return, sharpe_ratio, trade_count, etc.
                       
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if len(metrics_df) < self.min_sample_size:
            self.logger.warning(f"Insufficient data for pattern detection: {len(metrics_df)} samples")
            return patterns
        
        # Pattern 1: High Sharpe Strategies
        high_sharpe_pattern = self._detect_high_sharpe_pattern(metrics_df)
        if high_sharpe_pattern:
            patterns.append(high_sharpe_pattern)
        
        # Pattern 2: Low Drawdown Strategies
        low_drawdown_pattern = self._detect_low_drawdown_pattern(metrics_df)
        if low_drawdown_pattern:
            patterns.append(low_drawdown_pattern)
        
        # Pattern 3: High Activity vs Performance
        activity_pattern = self._detect_activity_pattern(metrics_df)
        if activity_pattern:
            patterns.append(activity_pattern)
        
        # Pattern 4: Consistent Winners
        consistency_pattern = self._detect_consistency_pattern(metrics_df)
        if consistency_pattern:
            patterns.append(consistency_pattern)
        
        self.logger.info(f"Detected {len(patterns)} patterns from {len(metrics_df)} samples")
        return patterns
    
    def _detect_high_sharpe_pattern(self, df: pd.DataFrame) -> Optional[Pattern]:
        """Detect high Sharpe ratio patterns."""
        
        # Define high Sharpe threshold (top 25% or minimum 1.5)
        sharpe_threshold = max(1.5, df['sharpe_ratio'].quantile(0.75))
        
        high_sharpe_df = df[df['sharpe_ratio'] >= sharpe_threshold]
        
        if len(high_sharpe_df) < self.min_sample_size:
            return None
        
        # Calculate success metrics
        avg_return = high_sharpe_df['total_return'].mean()
        success_rate = len(high_sharpe_df[high_sharpe_df['total_return'] > 0]) / len(high_sharpe_df)
        
        if success_rate < self.min_success_rate:
            return None
        
        # Analyze common characteristics
        avg_trades = high_sharpe_df['trade_count'].mean()
        avg_win_rate = high_sharpe_df['win_rate'].mean()
        
        return Pattern(
            pattern_type="high_sharpe",
            description=f"High Sharpe ratio strategies (>= {sharpe_threshold:.2f})",
            success_rate=success_rate,
            sample_count=len(high_sharpe_df),
            avg_return=avg_return,
            conditions={
                'min_sharpe_ratio': sharpe_threshold,
                'avg_trade_count': avg_trades,
                'avg_win_rate': avg_win_rate
            },
            correlation_ids=high_sharpe_df['correlation_id'].dropna().tolist()
        )
    
    def _detect_low_drawdown_pattern(self, df: pd.DataFrame) -> Optional[Pattern]:
        """Detect low drawdown patterns."""
        
        # Define low drawdown threshold (bottom 25% or maximum 5%)
        drawdown_threshold = min(5.0, df['max_drawdown'].quantile(0.25))
        
        low_dd_df = df[df['max_drawdown'] <= drawdown_threshold]
        
        if len(low_dd_df) < self.min_sample_size:
            return None
        
        # Calculate success metrics
        avg_return = low_dd_df['total_return'].mean()
        success_rate = len(low_dd_df[low_dd_df['total_return'] > 0]) / len(low_dd_df)
        
        if success_rate < self.min_success_rate:
            return None
        
        # Analyze characteristics
        avg_sharpe = low_dd_df['sharpe_ratio'].mean()
        avg_trades = low_dd_df['trade_count'].mean()
        
        return Pattern(
            pattern_type="low_drawdown",
            description=f"Low drawdown strategies (<= {drawdown_threshold:.1f}%)",
            success_rate=success_rate,
            sample_count=len(low_dd_df),
            avg_return=avg_return,
            conditions={
                'max_drawdown_threshold': drawdown_threshold,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_trade_count': avg_trades
            },
            correlation_ids=low_dd_df['correlation_id'].dropna().tolist()
        )
    
    def _detect_activity_pattern(self, df: pd.DataFrame) -> Optional[Pattern]:
        """Detect patterns related to trading activity level."""
        
        # Analyze relationship between trade count and performance
        high_activity_threshold = df['trade_count'].quantile(0.75)
        low_activity_threshold = df['trade_count'].quantile(0.25)
        
        high_activity_df = df[df['trade_count'] >= high_activity_threshold]
        low_activity_df = df[df['trade_count'] <= low_activity_threshold]
        
        if len(high_activity_df) < self.min_sample_size or len(low_activity_df) < self.min_sample_size:
            return None
        
        # Compare performance
        high_activity_return = high_activity_df['total_return'].mean()
        low_activity_return = low_activity_df['total_return'].mean()
        
        # Determine which activity level performs better
        if high_activity_return > low_activity_return:
            better_df = high_activity_df
            pattern_desc = f"High activity strategies (>= {high_activity_threshold:.0f} trades)"
            activity_type = "high"
            threshold = high_activity_threshold
        else:
            better_df = low_activity_df
            pattern_desc = f"Low activity strategies (<= {low_activity_threshold:.0f} trades)"
            activity_type = "low"
            threshold = low_activity_threshold
        
        success_rate = len(better_df[better_df['total_return'] > 0]) / len(better_df)
        
        if success_rate < self.min_success_rate:
            return None
        
        avg_return = better_df['total_return'].mean()
        avg_sharpe = better_df['sharpe_ratio'].mean()
        
        return Pattern(
            pattern_type=f"{activity_type}_activity",
            description=pattern_desc,
            success_rate=success_rate,
            sample_count=len(better_df),
            avg_return=avg_return,
            conditions={
                'activity_type': activity_type,
                'trade_count_threshold': threshold,
                'avg_sharpe_ratio': avg_sharpe
            },
            correlation_ids=better_df['correlation_id'].dropna().tolist()
        )
    
    def _detect_consistency_pattern(self, df: pd.DataFrame) -> Optional[Pattern]:
        """Detect patterns of consistent performance."""
        
        # Look for strategies with both positive returns AND decent win rates
        positive_return_df = df[df['total_return'] > 0]
        
        if len(positive_return_df) < self.min_sample_size:
            return None
        
        # Find strategies with above-average win rates
        avg_win_rate = positive_return_df['win_rate'].mean()
        consistent_df = positive_return_df[positive_return_df['win_rate'] >= avg_win_rate]
        
        if len(consistent_df) < self.min_sample_size:
            return None
        
        success_rate = 1.0  # By definition, all have positive returns
        avg_return = consistent_df['total_return'].mean()
        avg_sharpe = consistent_df['sharpe_ratio'].mean()
        avg_trades = consistent_df['trade_count'].mean()
        min_win_rate = consistent_df['win_rate'].min()
        
        return Pattern(
            pattern_type="consistent_winners",
            description=f"Consistent winning strategies (win rate >= {avg_win_rate:.1f}%)",
            success_rate=success_rate,
            sample_count=len(consistent_df),
            avg_return=avg_return,
            conditions={
                'min_win_rate': avg_win_rate,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_trade_count': avg_trades,
                'lowest_win_rate': min_win_rate
            },
            correlation_ids=consistent_df['correlation_id'].dropna().tolist()
        )
    
    def analyze_pattern_overlap(self, patterns: List[Pattern]) -> Dict[str, Any]:
        """
        Analyze overlap between patterns to find compound patterns.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Analysis of pattern interactions
        """
        if len(patterns) < 2:
            return {'message': 'Need at least 2 patterns for overlap analysis'}
        
        overlap_analysis = {}
        
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                # Find correlation IDs that appear in both patterns
                overlap_ids = set(pattern1.correlation_ids) & set(pattern2.correlation_ids)
                
                if overlap_ids:
                    overlap_key = f"{pattern1.pattern_type}_+_{pattern2.pattern_type}"
                    overlap_analysis[overlap_key] = {
                        'pattern_1': pattern1.pattern_type,
                        'pattern_2': pattern2.pattern_type,
                        'overlap_count': len(overlap_ids),
                        'pattern_1_total': pattern1.sample_count,
                        'pattern_2_total': pattern2.sample_count,
                        'overlap_percentage_1': len(overlap_ids) / pattern1.sample_count * 100,
                        'overlap_percentage_2': len(overlap_ids) / pattern2.sample_count * 100,
                        'overlapping_correlation_ids': list(overlap_ids)
                    }
        
        return overlap_analysis
    
    def generate_pattern_summary(self, patterns: List[Pattern]) -> str:
        """Generate human-readable summary of detected patterns."""
        
        if not patterns:
            return "No significant patterns detected in the current dataset."
        
        summary = ["=== PATTERN DETECTION SUMMARY ===\n"]
        
        for i, pattern in enumerate(patterns, 1):
            summary.append(f"{i}. {pattern.description}")
            summary.append(f"   Success Rate: {pattern.success_rate:.1%}")
            summary.append(f"   Sample Size: {pattern.sample_count}")
            summary.append(f"   Avg Return: {pattern.avg_return:.2f}%")
            summary.append(f"   Event Traces Available: {len(pattern.correlation_ids)} correlation IDs")
            summary.append("")
        
        summary.append("=== RECOMMENDATIONS ===")
        
        # Sort patterns by a composite score
        scored_patterns = []
        for pattern in patterns:
            # Score = success_rate * log(sample_size) * avg_return
            score = pattern.success_rate * np.log(pattern.sample_count) * max(0.1, pattern.avg_return)
            scored_patterns.append((pattern, score))
        
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        if scored_patterns:
            best_pattern = scored_patterns[0][0]
            summary.append(f"1. Focus on '{best_pattern.pattern_type}' patterns")
            summary.append(f"   - Highest composite score with {best_pattern.success_rate:.1%} success rate")
            summary.append(f"   - Use correlation IDs for deep-dive event analysis")
            summary.append("")
        
        if len(patterns) > 1:
            summary.append("2. Investigate pattern combinations")
            summary.append("   - Look for strategies that match multiple patterns")
            summary.append("   - Use overlap analysis to find compound patterns")
        
        return "\n".join(summary)