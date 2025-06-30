"""
Trace Analysis for ADMF-PC

Interactive analysis of strategy traces with focus on pattern discovery
and exploratory workflows. Designed for Jupyter notebook usage.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TraceSession:
    """Represents a trace analysis session."""
    workspace_path: Path
    conn: duckdb.DuckDBPyConnection
    metadata: Dict[str, Any]
    strategy_map: Dict[int, Dict[str, Any]]
    
    @property
    def num_strategies(self) -> int:
        return len(self.strategy_map)
    
    @property
    def total_signals(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]


class TraceAnalysis:
    """
    Main interface for trace analysis.
    
    Provides interactive methods for analyzing sparse signal traces
    stored in parquet format.
    
    Example:
        >>> ta = TraceAnalysis("config/keltner/results/latest")
        >>> ta.summary()  # Get quick overview
        >>> ta.compare_filters()  # Compare filtered vs unfiltered
        >>> ta.find_patterns(min_occurrences=10)  # Discover patterns
    """
    
    def __init__(self, workspace_path: Union[str, Path]):
        """
        Initialize trace analysis for a workspace.
        
        Args:
            workspace_path: Path to results directory containing traces
        """
        self.workspace_path = Path(workspace_path)
        self.conn = None
        self.session = None
        self._setup_session()
    
    def _setup_session(self):
        """Set up DuckDB connection and load metadata."""
        # Initialize DuckDB
        self.conn = duckdb.connect(':memory:')
        
        # Enable progress bar for long queries
        self.conn.execute("SET enable_progress_bar = true")
        
        # Load metadata
        metadata_path = self.workspace_path / "metadata_enhanced.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Create strategy map for easy lookup
        strategy_map = {}
        for strategy in metadata.get('strategies', []):
            strategy_map[strategy['id']] = strategy
        
        # Create session
        self.session = TraceSession(
            workspace_path=self.workspace_path,
            conn=self.conn,
            metadata=metadata,
            strategy_map=strategy_map
        )
        
        # Load all parquet files
        self._load_traces()
    
    def _load_traces(self):
        """Load all trace parquet files into DuckDB."""
        traces_dir = self.workspace_path / "traces"
        if not traces_dir.exists():
            logger.warning(f"No traces directory found at {traces_dir}")
            return
        
        # Find all parquet files
        parquet_files = list(traces_dir.glob("*.parquet"))
        logger.info(f"Found {len(parquet_files)} trace files")
        
        if not parquet_files:
            return
        
        # Create consolidated signals view
        # Sparse format has columns: idx (bar index), val (signal value), px (price)
        first_file = parquet_files[0]
        strategy_id = int(first_file.stem.split('_')[1])
        
        query = f"""
        CREATE OR REPLACE VIEW signals AS
        SELECT 
            {strategy_id} as strategy_id,
            idx as bar_idx,
            val as signal_value,
            px as price
        FROM read_parquet('{first_file}')
        """
        
        # Add remaining files
        for pf in parquet_files[1:]:
            strategy_id = int(pf.stem.split('_')[1])
            query += f"""
            UNION ALL
            SELECT 
                {strategy_id} as strategy_id,
                idx as bar_idx,
                val as signal_value,
                px as price
            FROM read_parquet('{pf}')
            """
        
        self.conn.execute(query)
        
        # Create additional useful views
        self._create_analysis_views()
    
    def _create_analysis_views(self):
        """Create helpful views for analysis."""
        # Signal transitions view
        self.conn.execute("""
        CREATE OR REPLACE VIEW signal_transitions AS
        WITH lagged AS (
            SELECT 
                strategy_id,
                bar_idx,
                signal_value,
                price,
                LAG(signal_value, 1, 0) OVER (PARTITION BY strategy_id ORDER BY bar_idx) as prev_signal
            FROM signals
        )
        SELECT 
            strategy_id,
            bar_idx,
            signal_value,
            prev_signal,
            price,
            CASE 
                WHEN signal_value > 0 AND prev_signal <= 0 THEN 'long_entry'
                WHEN signal_value < 0 AND prev_signal >= 0 THEN 'short_entry'
                WHEN signal_value = 0 AND prev_signal != 0 THEN 'exit'
                ELSE 'hold'
            END as transition_type
        FROM lagged
        WHERE signal_value != prev_signal
        """)
        
        # Strategy summary view
        self.conn.execute("""
        CREATE OR REPLACE VIEW strategy_summary AS
        SELECT 
            strategy_id,
            COUNT(*) as total_signals,
            SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END) as long_signals,
            SUM(CASE WHEN signal_value < 0 THEN 1 ELSE 0 END) as short_signals,
            MIN(bar_idx) as first_signal_bar,
            MAX(bar_idx) as last_signal_bar
        FROM signals
        GROUP BY strategy_id
        """)
    
    def summary(self) -> pd.DataFrame:
        """
        Get a quick summary of all strategies.
        
        Returns:
            DataFrame with strategy summaries including signal counts and filters
        """
        # Get signal summary
        summary_df = self.conn.execute("SELECT * FROM strategy_summary").df()
        
        # Enhance with metadata
        enhanced_rows = []
        for _, row in summary_df.iterrows():
            strategy = self.session.strategy_map.get(row['strategy_id'], {})
            
            enhanced_row = {
                'strategy_id': row['strategy_id'],
                'strategy_name': strategy.get('name', 'unknown'),
                'filter': str(strategy.get('filter', 'none')),
                'total_signals': row['total_signals'],
                'long_signals': row['long_signals'],
                'short_signals': row['short_signals'],
                'signal_bars': row['last_signal_bar'] - row['first_signal_bar'] + 1
            }
            
            # Add key parameters
            params = strategy.get('params', {})
            for key in ['period', 'multiplier', 'threshold']:
                if key in params:
                    enhanced_row[key] = params[key]
            
            enhanced_rows.append(enhanced_row)
        
        return pd.DataFrame(enhanced_rows)
    
    def compare_filters(self, baseline_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Compare filtered strategies against baseline.
        
        Args:
            baseline_filter: Filter expression for baseline (default: None/null)
            
        Returns:
            DataFrame showing filter effectiveness
        """
        summary = self.summary()
        
        # Find baseline strategies
        if baseline_filter is None:
            baseline_mask = summary['filter'].str.lower().isin(['none', 'null', '[]'])
        else:
            baseline_mask = summary['filter'] == baseline_filter
        
        baseline = summary[baseline_mask]
        filtered = summary[~baseline_mask]
        
        if baseline.empty:
            logger.warning("No baseline strategies found")
            return pd.DataFrame()
        
        # Group by filter type and compare
        results = []
        for filter_type in filtered['filter'].unique():
            filter_strategies = filtered[filtered['filter'] == filter_type]
            
            # Calculate average reduction
            baseline_avg = baseline['total_signals'].mean()
            filter_avg = filter_strategies['total_signals'].mean()
            reduction_pct = (baseline_avg - filter_avg) / baseline_avg * 100
            
            results.append({
                'filter': filter_type,
                'strategies': len(filter_strategies),
                'avg_signals': filter_avg,
                'baseline_avg': baseline_avg,
                'reduction_pct': reduction_pct,
                'min_signals': filter_strategies['total_signals'].min(),
                'max_signals': filter_strategies['total_signals'].max()
            })
        
        return pd.DataFrame(results).sort_values('reduction_pct', ascending=False)
    
    def find_patterns(self, min_occurrences: int = 5, 
                     window_size: int = 20) -> Dict[str, Any]:
        """
        Discover common signal patterns across strategies.
        
        Args:
            min_occurrences: Minimum times pattern must appear
            window_size: Size of pattern window in bars
            
        Returns:
            Dictionary of discovered patterns
        """
        # Get signal sequences for pattern matching
        query = f"""
        WITH signal_sequences AS (
            SELECT 
                strategy_id,
                bar_idx,
                signal_value,
                STRING_AGG(
                    CAST(signal_value AS VARCHAR), 
                    ',' 
                    ORDER BY bar_idx
                ) OVER (
                    PARTITION BY strategy_id 
                    ORDER BY bar_idx 
                    ROWS BETWEEN {window_size} PRECEDING AND CURRENT ROW
                ) as pattern
            FROM signals
        )
        SELECT 
            pattern,
            COUNT(DISTINCT strategy_id) as num_strategies,
            COUNT(*) as occurrences,
            AVG(signal_value) as avg_signal_strength
        FROM signal_sequences
        WHERE LENGTH(pattern) > 10  -- Non-trivial patterns
        GROUP BY pattern
        HAVING COUNT(*) >= {min_occurrences}
        ORDER BY occurrences DESC
        LIMIT 50
        """
        
        patterns_df = self.conn.execute(query).df()
        
        # Analyze patterns
        patterns = {
            'frequent_patterns': patterns_df.to_dict('records'),
            'summary': {
                'total_patterns': len(patterns_df),
                'avg_occurrences': patterns_df['occurrences'].mean() if not patterns_df.empty else 0,
                'strategies_with_patterns': patterns_df['num_strategies'].sum() if not patterns_df.empty else 0
            }
        }
        
        return patterns
    
    def analyze_transitions(self, strategy_id: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze signal transitions (entries/exits).
        
        Args:
            strategy_id: Specific strategy to analyze (None for all)
            
        Returns:
            DataFrame with transition analysis
        """
        where_clause = f"WHERE strategy_id = {strategy_id}" if strategy_id else ""
        
        query = f"""
        SELECT 
            transition_type,
            COUNT(*) as count,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price,
            STDDEV(price) as price_stddev
        FROM signal_transitions
        {where_clause}
        GROUP BY transition_type
        ORDER BY count DESC
        """
        
        return self.conn.execute(query).df()
    
    def get_trades(self, strategy_id: Optional[int] = None,
                   min_duration: int = 0) -> pd.DataFrame:
        """
        Extract trade-like sequences from signals.
        
        Args:
            strategy_id: Specific strategy (None for all)
            min_duration: Minimum bars to hold position
            
        Returns:
            DataFrame with trade information
        """
        where_clause = f"AND strategy_id = {strategy_id}" if strategy_id else ""
        
        query = f"""
        WITH entries AS (
            SELECT 
                strategy_id,
                bar_idx as entry_bar,
                price as entry_price,
                signal_value as entry_signal,
                transition_type,
                ROW_NUMBER() OVER (PARTITION BY strategy_id ORDER BY bar_idx) as entry_num
            FROM signal_transitions
            WHERE transition_type IN ('long_entry', 'short_entry')
            {where_clause}
        ),
        exits AS (
            SELECT 
                strategy_id,
                bar_idx as exit_bar,
                price as exit_price,
                ROW_NUMBER() OVER (PARTITION BY strategy_id ORDER BY bar_idx) as exit_num
            FROM signal_transitions
            WHERE transition_type = 'exit'
            {where_clause}
        )
        SELECT 
            e.strategy_id,
            e.entry_bar,
            e.entry_price,
            e.entry_signal,
            x.exit_bar,
            x.exit_price,
            x.exit_bar - e.entry_bar as duration_bars,
            CASE 
                WHEN e.entry_signal > 0 THEN (x.exit_price - e.entry_price) / e.entry_price
                ELSE (e.entry_price - x.exit_price) / e.entry_price
            END as return_pct
        FROM entries e
        LEFT JOIN exits x 
            ON e.strategy_id = x.strategy_id 
            AND x.exit_num = e.entry_num
        WHERE x.exit_bar - e.entry_bar >= {min_duration}
        ORDER BY e.strategy_id, e.entry_bar
        """
        
        return self.conn.execute(query).df()
    
    def performance_by_filter(self) -> pd.DataFrame:
        """
        Calculate hypothetical performance grouped by filter type.
        
        Returns:
            DataFrame with performance metrics by filter
        """
        trades_df = self.get_trades()
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # Add filter information
        trades_df['filter'] = trades_df['strategy_id'].map(
            lambda sid: self.session.strategy_map.get(sid, {}).get('filter', 'none')
        )
        
        # Calculate metrics by filter
        results = []
        for filter_type in trades_df['filter'].unique():
            filter_trades = trades_df[trades_df['filter'] == str(filter_type)]
            
            if not filter_trades.empty:
                results.append({
                    'filter': filter_type,
                    'num_trades': len(filter_trades),
                    'avg_return': filter_trades['return_pct'].mean(),
                    'win_rate': (filter_trades['return_pct'] > 0).mean(),
                    'avg_duration': filter_trades['duration_bars'].mean(),
                    'sharpe': filter_trades['return_pct'].mean() / filter_trades['return_pct'].std() 
                              if filter_trades['return_pct'].std() > 0 else 0
                })
        
        return pd.DataFrame(results).sort_values('sharpe', ascending=False)
    
    def export_for_mining(self, output_path: Optional[Path] = None) -> Path:
        """
        Export consolidated data for external analysis.
        
        Args:
            output_path: Where to save (default: workspace/analysis/)
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = self.workspace_path / "analysis" / "consolidated_traces.parquet"
        
        output_path.parent.mkdir(exist_ok=True)
        
        # Export enriched signals
        query = """
        SELECT 
            s.*,
            st.transition_type,
            ss.total_signals as strategy_total_signals
        FROM signals s
        LEFT JOIN signal_transitions st 
            ON s.strategy_id = st.strategy_id 
            AND s.bar_idx = st.bar_idx
        LEFT JOIN strategy_summary ss
            ON s.strategy_id = ss.strategy_id
        ORDER BY s.strategy_id, s.bar_idx
        """
        
        export_df = self.conn.execute(query).df()
        
        # Add metadata
        export_df['filter'] = export_df['strategy_id'].map(
            lambda sid: str(self.session.strategy_map.get(sid, {}).get('filter', 'none'))
        )
        export_df['strategy_name'] = export_df['strategy_id'].map(
            lambda sid: self.session.strategy_map.get(sid, {}).get('name', 'unknown')
        )
        
        # Save
        export_df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(export_df)} records to {output_path}")
        
        return output_path
    
    def close(self):
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __repr__(self):
        if self.session:
            return (f"TraceAnalysis(workspace='{self.workspace_path.name}', "
                   f"strategies={self.session.num_strategies}, "
                   f"signals={self.session.total_signals})")
        return f"TraceAnalysis(workspace='{self.workspace_path}')"