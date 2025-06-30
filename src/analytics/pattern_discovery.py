"""
Pattern Discovery for ADMF-PC

Tools for finding, validating, and saving trading patterns.
Builds institutional knowledge over time.
"""

import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TradingPattern:
    """Represents a discovered trading pattern."""
    pattern_id: str
    name: str
    description: str
    query: str
    discovery_date: str
    performance_metrics: Dict[str, float]
    parameters: Dict[str, Any]
    validation_results: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingPattern':
        """Create from dictionary."""
        return cls(**data)


class PatternLibrary:
    """
    Manages a library of discovered trading patterns.
    
    Patterns are stored in YAML for human readability and git tracking.
    """
    
    def __init__(self, library_path: Optional[Path] = None):
        """
        Initialize pattern library.
        
        Args:
            library_path: Path to patterns file (default: analytics/saved_patterns/patterns.yaml)
        """
        if library_path is None:
            library_path = Path(__file__).parent / "saved_patterns" / "patterns.yaml"
        
        self.library_path = Path(library_path)
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, TradingPattern]:
        """Load patterns from disk."""
        patterns = {}
        
        if self.library_path.exists():
            with open(self.library_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                
            for pattern_id, pattern_data in data.items():
                patterns[pattern_id] = TradingPattern.from_dict(pattern_data)
        
        return patterns
    
    def save(self):
        """Save patterns to disk."""
        data = {
            pid: pattern.to_dict() 
            for pid, pattern in self.patterns.items()
        }
        
        with open(self.library_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def add_pattern(self, pattern: TradingPattern) -> str:
        """Add a new pattern to the library."""
        self.patterns[pattern.pattern_id] = pattern
        self.save()
        logger.info(f"Added pattern: {pattern.name}")
        return pattern.pattern_id
    
    def get_pattern(self, pattern_id: str) -> Optional[TradingPattern]:
        """Get a pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def search(self, tags: List[str] = None, 
               min_sharpe: float = None,
               text: str = None) -> List[TradingPattern]:
        """
        Search patterns by criteria.
        
        Args:
            tags: Filter by tags
            min_sharpe: Minimum Sharpe ratio
            text: Search in name/description
            
        Returns:
            List of matching patterns
        """
        results = []
        
        for pattern in self.patterns.values():
            # Tag filter
            if tags and pattern.tags:
                if not any(tag in pattern.tags for tag in tags):
                    continue
            
            # Performance filter
            if min_sharpe is not None:
                sharpe = pattern.performance_metrics.get('sharpe_ratio', 0)
                if sharpe < min_sharpe:
                    continue
            
            # Text search
            if text:
                text_lower = text.lower()
                if (text_lower not in pattern.name.lower() and 
                    text_lower not in pattern.description.lower()):
                    continue
            
            results.append(pattern)
        
        return results


class PatternDiscovery:
    """
    Interactive pattern discovery tools.
    
    Helps find patterns in trading data and validate them.
    """
    
    def __init__(self, trace_analysis):
        """
        Initialize with TraceAnalysis instance.
        
        Args:
            trace_analysis: TraceAnalysis instance with loaded data
        """
        self.ta = trace_analysis
        self.library = PatternLibrary()
        self.discovered_patterns = []
    
    def find_signal_sequences(self, min_frequency: float = 0.01,
                            window_size: int = 10) -> pd.DataFrame:
        """
        Find common signal sequences.
        
        Args:
            min_frequency: Minimum frequency (0-1) to be considered common
            window_size: Size of sequence window
            
        Returns:
            DataFrame of common sequences
        """
        query = f"""
        WITH signal_sequences AS (
            SELECT 
                strategy_id,
                bar_idx,
                STRING_AGG(
                    CASE 
                        WHEN signal_value > 0 THEN 'L'
                        WHEN signal_value < 0 THEN 'S'
                        ELSE 'N'
                    END,
                    ''
                ) OVER (
                    PARTITION BY strategy_id 
                    ORDER BY bar_idx 
                    ROWS BETWEEN {window_size-1} PRECEDING AND CURRENT ROW
                ) as sequence,
                LEAD(signal_value, 1) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as next_signal,
                LEAD(price, 1) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as next_price,
                price
            FROM signals
        ),
        sequence_outcomes AS (
            SELECT 
                sequence,
                COUNT(*) as occurrences,
                COUNT(DISTINCT strategy_id) as strategies,
                AVG(CASE 
                    WHEN next_signal = 0 AND signal_value != 0 
                    THEN (next_price - price) / price 
                    ELSE NULL 
                END) as avg_return,
                SUM(CASE WHEN next_signal > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as next_long_prob,
                SUM(CASE WHEN next_signal < 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as next_short_prob
            FROM signal_sequences
            WHERE LENGTH(sequence) = {window_size}
            GROUP BY sequence
        )
        SELECT *
        FROM sequence_outcomes
        WHERE occurrences::FLOAT / (SELECT COUNT(*) FROM signal_sequences WHERE LENGTH(sequence) = {window_size}) >= {min_frequency}
        ORDER BY occurrences DESC
        """
        
        return self.ta.conn.execute(query).df()
    
    def find_profitable_conditions(self, min_trades: int = 10,
                                 min_sharpe: float = 1.0) -> pd.DataFrame:
        """
        Find market conditions where strategies are most profitable.
        
        Args:
            min_trades: Minimum trades in condition
            min_sharpe: Minimum Sharpe ratio
            
        Returns:
            DataFrame of profitable conditions
        """
        query = f"""
        WITH market_conditions AS (
            SELECT 
                strategy_id,
                bar_idx,
                price,
                signal_value,
                -- Volatility condition
                CASE 
                    WHEN STDDEV(price) OVER (ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) 
                         > PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY STDDEV(price) OVER (ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)) OVER ()
                    THEN 'high_vol'
                    WHEN STDDEV(price) OVER (ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) 
                         < PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY STDDEV(price) OVER (ORDER BY bar_idx ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)) OVER ()
                    THEN 'low_vol'
                    ELSE 'normal_vol'
                END as volatility,
                -- Trend condition
                CASE 
                    WHEN (price - AVG(price) OVER (ORDER BY bar_idx ROWS BETWEEN 50 PRECEDING AND CURRENT ROW)) 
                         / AVG(price) OVER (ORDER BY bar_idx ROWS BETWEEN 50 PRECEDING AND CURRENT ROW) > 0.02
                    THEN 'uptrend'
                    WHEN (price - AVG(price) OVER (ORDER BY bar_idx ROWS BETWEEN 50 PRECEDING AND CURRENT ROW)) 
                         / AVG(price) OVER (ORDER BY bar_idx ROWS BETWEEN 50 PRECEDING AND CURRENT ROW) < -0.02
                    THEN 'downtrend'
                    ELSE 'sideways'
                END as trend
            FROM signals
        ),
        condition_performance AS (
            SELECT 
                volatility,
                trend,
                strategy_id,
                COUNT(*) as trades,
                AVG(
                    CASE 
                        WHEN signal_value = 0 AND LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) != 0 
                        THEN (price - LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx)) / LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx)
                        ELSE NULL
                    END
                ) as avg_return,
                STDDEV(
                    CASE 
                        WHEN signal_value = 0 AND LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) != 0 
                        THEN (price - LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx)) / LAG(price) OVER (PARTITION BY strategy_id ORDER BY bar_idx)
                        ELSE NULL
                    END
                ) as return_std
            FROM market_conditions
            WHERE signal_value = 0 AND LAG(signal_value) OVER (PARTITION BY strategy_id ORDER BY bar_idx) != 0
            GROUP BY volatility, trend, strategy_id
        )
        SELECT 
            volatility,
            trend,
            COUNT(DISTINCT strategy_id) as strategies,
            SUM(trades) as total_trades,
            AVG(avg_return) as avg_return,
            AVG(avg_return) / NULLIF(AVG(return_std), 0) * SQRT(252) as sharpe_ratio
        FROM condition_performance
        GROUP BY volatility, trend
        HAVING SUM(trades) >= {min_trades} AND AVG(avg_return) / NULLIF(AVG(return_std), 0) * SQRT(252) >= {min_sharpe}
        ORDER BY sharpe_ratio DESC
        """
        
        return self.ta.conn.execute(query).df()
    
    def discover_parameter_edges(self, param_name: str = 'period',
                               percentile_cutoff: float = 0.9) -> Dict[str, Any]:
        """
        Find parameter values that produce exceptional results.
        
        Args:
            param_name: Parameter to analyze
            percentile_cutoff: Performance percentile to consider exceptional
            
        Returns:
            Dictionary with edge cases and analysis
        """
        # This would need to join with metadata to get parameters
        # For now, return a template structure
        return {
            'parameter': param_name,
            'exceptional_values': [],
            'analysis': 'Requires metadata integration',
            'recommendation': 'Focus testing on edge values'
        }
    
    def create_pattern(self, name: str, description: str,
                      query: str, tags: List[str] = None) -> TradingPattern:
        """
        Create a new pattern from a query.
        
        Args:
            name: Pattern name
            description: What the pattern captures
            query: SQL query that identifies the pattern
            tags: Optional tags for categorization
            
        Returns:
            New TradingPattern instance
        """
        # Run the query to get performance metrics
        try:
            results = self.ta.conn.execute(query).df()
            
            if results.empty:
                metrics = {'status': 'no_results'}
            else:
                # Extract key metrics from results
                metrics = {
                    'result_count': len(results),
                    'columns': list(results.columns)
                }
                
                # Try to extract common metrics if they exist
                for col in ['sharpe_ratio', 'avg_return', 'win_rate', 'trades']:
                    if col in results.columns:
                        metrics[col] = float(results[col].mean())
        
        except Exception as e:
            logger.error(f"Error running pattern query: {e}")
            metrics = {'status': 'error', 'error': str(e)}
        
        pattern = TradingPattern(
            pattern_id=f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            description=description,
            query=query,
            discovery_date=datetime.now().isoformat(),
            performance_metrics=metrics,
            parameters={},
            tags=tags or []
        )
        
        return pattern
    
    def validate_pattern(self, pattern: TradingPattern,
                        test_workspace: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validate a pattern on out-of-sample data.
        
        Args:
            pattern: Pattern to validate
            test_workspace: Optional different workspace for validation
            
        Returns:
            Validation results
        """
        validation_results = {
            'pattern_id': pattern.pattern_id,
            'validation_date': datetime.now().isoformat(),
            'original_metrics': pattern.performance_metrics
        }
        
        try:
            if test_workspace:
                # Would need to load different workspace
                validation_results['status'] = 'requires_workspace_switch'
            else:
                # Run pattern query on current data
                results = self.ta.conn.execute(pattern.query).df()
                
                validation_results['status'] = 'success'
                validation_results['validation_metrics'] = {
                    'result_count': len(results)
                }
                
                # Compare with original metrics
                for metric in ['sharpe_ratio', 'avg_return', 'win_rate']:
                    if metric in pattern.performance_metrics and not results.empty:
                        if metric in results.columns:
                            new_val = float(results[metric].mean())
                            orig_val = pattern.performance_metrics[metric]
                            validation_results['validation_metrics'][metric] = new_val
                            validation_results['validation_metrics'][f'{metric}_change'] = (
                                (new_val - orig_val) / orig_val if orig_val != 0 else 0
                            )
        
        except Exception as e:
            validation_results['status'] = 'error'
            validation_results['error'] = str(e)
        
        return validation_results
    
    def save_pattern(self, pattern: TradingPattern) -> str:
        """
        Save a pattern to the library.
        
        Args:
            pattern: Pattern to save
            
        Returns:
            Pattern ID
        """
        return self.library.add_pattern(pattern)
    
    def suggest_explorations(self) -> List[Dict[str, str]]:
        """
        Suggest interesting explorations based on current data.
        
        Returns:
            List of exploration suggestions
        """
        suggestions = []
        
        # Check for basic patterns
        summary = self.ta.summary()
        
        if not summary.empty:
            # High signal variation
            signal_std = summary['total_signals'].std()
            signal_mean = summary['total_signals'].mean()
            if signal_std / signal_mean > 0.5:
                suggestions.append({
                    'title': 'High Signal Variation',
                    'description': 'Large differences in signal counts between strategies',
                    'action': 'Investigate why some strategies signal much more than others'
                })
            
            # Filter effectiveness
            if 'filter' in summary.columns:
                unique_filters = summary['filter'].nunique()
                if unique_filters > 1:
                    suggestions.append({
                        'title': 'Compare Filter Effectiveness',
                        'description': f'Found {unique_filters} different filter types',
                        'action': 'Run compare_filters() to see which work best'
                    })
            
            # Parameter patterns
            for param in ['period', 'multiplier', 'threshold']:
                if param in summary.columns:
                    unique_vals = summary[param].nunique()
                    if unique_vals > 3:
                        suggestions.append({
                            'title': f'Analyze {param} Impact',
                            'description': f'Found {unique_vals} different {param} values',
                            'action': f'Explore how {param} affects performance'
                        })
        
        # Always suggest pattern discovery
        suggestions.append({
            'title': 'Discover Signal Patterns',
            'description': 'Find common signal sequences that precede profits',
            'action': 'Run find_signal_sequences() to discover patterns'
        })
        
        return suggestions