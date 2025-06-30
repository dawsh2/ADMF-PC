"""
Global Trace Store

Provides unified access to traces stored in the global traces/ directory.
Enables cross-run analysis and strategy deduplication.
"""

import pandas as pd
import duckdb
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import json

logger = logging.getLogger(__name__)


class TraceStore:
    """
    Global trace store for accessing strategy traces across all runs.
    
    Features:
    - Query strategies by parameters across all runs
    - Load traces by strategy hash
    - Find duplicate strategies
    - Performance analytics integration
    """
    
    def __init__(self, traces_root: Optional[str] = None):
        """
        Initialize trace store.
        
        Args:
            traces_root: Path to global traces directory (defaults to project root/traces)
        """
        if traces_root:
            self._traces_root = Path(traces_root)
        else:
            # Find project root by looking for marker files
            current = Path.cwd()
            while current != current.parent:
                if (current / '.git').exists() or (current / 'pyproject.toml').exists():
                    self._traces_root = current / 'traces'
                    break
                current = current.parent
            else:
                # Fallback to cwd/traces
                self._traces_root = Path.cwd() / 'traces'
        
        if not self._traces_root.exists():
            logger.warning(f"Global traces directory not found: {self._traces_root}")
            self._traces_root.mkdir(parents=True, exist_ok=True)
        
        self._strategy_index_path = self._traces_root / 'strategy_index.parquet'
        self._strategy_index: Optional[pd.DataFrame] = None
        
        logger.info(f"TraceStore initialized with root: {self._traces_root}")
        
        # Load strategy index if available
        if self._strategy_index_path.exists():
            self._load_strategy_index()
    
    def _load_strategy_index(self):
        """Load the global strategy index."""
        try:
            self._strategy_index = pd.read_parquet(self._strategy_index_path)
            logger.info(f"Loaded strategy index with {len(self._strategy_index)} entries")
        except Exception as e:
            logger.error(f"Failed to load strategy index: {e}")
            self._strategy_index = None
    
    def refresh_index(self):
        """Refresh the strategy index from disk."""
        self._load_strategy_index()
    
    @property
    def has_index(self) -> bool:
        """Check if strategy index is available."""
        return self._strategy_index is not None
    
    def list_strategies(self, strategy_type: Optional[str] = None) -> pd.DataFrame:
        """
        List all available strategies.
        
        Args:
            strategy_type: Filter by strategy type (e.g., 'bollinger_bands')
            
        Returns:
            DataFrame with strategy information
        """
        if self._strategy_index is None:
            return pd.DataFrame()
        
        df = self._strategy_index.copy()
        
        if strategy_type:
            df = df[df['strategy_type'] == strategy_type]
        
        return df
    
    def find_strategy(self, **params) -> pd.DataFrame:
        """
        Find strategies matching specific parameters.
        
        Example:
            store.find_strategy(strategy_type='bollinger_bands', period=20, std_dev=2.0)
            
        Returns:
            DataFrame with matching strategies
        """
        if self._strategy_index is None:
            return pd.DataFrame()
        
        df = self._strategy_index.copy()
        
        # Filter by each parameter
        for param, value in params.items():
            if param in df.columns:
                df = df[df[param] == value]
        
        return df
    
    def load_trace(self, strategy_hash: str) -> Optional[pd.DataFrame]:
        """
        Load trace data for a specific strategy hash.
        
        Args:
            strategy_hash: Strategy hash to load
            
        Returns:
            DataFrame with trace data or None if not found
        """
        if self._strategy_index is None:
            return None
        
        # Find trace path for this hash
        matches = self._strategy_index[self._strategy_index['strategy_hash'] == strategy_hash]
        if matches.empty:
            logger.warning(f"No trace found for hash: {strategy_hash}")
            return None
        
        trace_path = matches.iloc[0]['trace_path']
        full_path = self._traces_root.parent / trace_path  # trace_path is relative to project root
        
        if not full_path.exists():
            logger.error(f"Trace file not found: {full_path}")
            return None
        
        try:
            return pd.read_parquet(full_path)
        except Exception as e:
            logger.error(f"Failed to load trace: {e}")
            return None
    
    def load_traces(self, strategy_hashes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple traces by their hashes.
        
        Args:
            strategy_hashes: List of strategy hashes
            
        Returns:
            Dict mapping hash to trace DataFrame
        """
        traces = {}
        for hash_val in strategy_hashes:
            trace = self.load_trace(hash_val)
            if trace is not None:
                traces[hash_val] = trace
        return traces
    
    def query(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query against the strategy index.
        
        Example:
            store.query("SELECT * FROM strategy_index WHERE sharpe_ratio > 1.5")
            
        Args:
            sql: SQL query (strategy_index is available as a table)
            
        Returns:
            Query results as DataFrame
        """
        if self._strategy_index is None:
            return pd.DataFrame()
        
        conn = duckdb.connect(':memory:')
        
        # Register the strategy index
        conn.register('strategy_index', self._strategy_index)
        
        try:
            result = conn.execute(sql).df()
            return result
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def find_duplicates(self) -> pd.DataFrame:
        """
        Find strategies that appear in multiple runs.
        
        Returns:
            DataFrame with duplicate strategies and their counts
        """
        if self._strategy_index is None:
            return pd.DataFrame()
        
        # Group by hash and count occurrences
        duplicates = self._strategy_index.groupby('strategy_hash').agg({
            'strategy_type': 'first',
            'trace_path': 'count'
        }).rename(columns={'trace_path': 'run_count'})
        
        # Filter to only duplicates
        duplicates = duplicates[duplicates['run_count'] > 1]
        duplicates = duplicates.sort_values('run_count', ascending=False)
        
        return duplicates.reset_index()
    
    def get_trace_path(self, strategy_hash: str) -> Optional[Path]:
        """
        Get the file path for a strategy hash.
        
        Args:
            strategy_hash: Strategy hash
            
        Returns:
            Path to trace file or None
        """
        if self._strategy_index is None:
            return None
        
        matches = self._strategy_index[self._strategy_index['strategy_hash'] == strategy_hash]
        if matches.empty:
            return None
        
        trace_path = matches.iloc[0]['trace_path']
        return self._traces_root.parent / trace_path
    
    def analyze_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """
        Analyze parameter distributions for a strategy type.
        
        Args:
            strategy_type: Type of strategy to analyze
            
        Returns:
            Dict with parameter statistics
        """
        if self._strategy_index is None:
            return {}
        
        df = self._strategy_index[self._strategy_index['strategy_type'] == strategy_type]
        
        if df.empty:
            return {}
        
        # Get parameter columns (exclude metadata columns)
        meta_cols = ['strategy_id', 'strategy_hash', 'strategy_type', 'symbol', 
                     'timeframe', 'constraints', 'trace_path', 'first_seen']
        param_cols = [col for col in df.columns if col not in meta_cols]
        
        stats = {}
        for param in param_cols:
            if param in df.columns:
                param_data = df[param].dropna()
                if not param_data.empty:
                    stats[param] = {
                        'unique_values': param_data.nunique(),
                        'min': param_data.min() if pd.api.types.is_numeric_dtype(param_data) else None,
                        'max': param_data.max() if pd.api.types.is_numeric_dtype(param_data) else None,
                        'mean': param_data.mean() if pd.api.types.is_numeric_dtype(param_data) else None,
                        'values': param_data.value_counts().to_dict()
                    }
        
        return stats
    
    def load_workspace_traces(self, workspace_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Load all traces referenced by a specific workspace.
        
        Args:
            workspace_path: Path to workspace directory
            
        Returns:
            Dict mapping strategy_id to trace DataFrame
        """
        workspace_path = Path(workspace_path)
        workspace_index = workspace_path / 'strategy_index.parquet'
        
        if not workspace_index.exists():
            logger.warning(f"No strategy index found in workspace: {workspace_path}")
            return {}
        
        try:
            # Load workspace index
            ws_index = pd.read_parquet(workspace_index)
            
            # Load each unique trace
            traces = {}
            for _, row in ws_index.iterrows():
                strategy_id = row['strategy_id']
                strategy_hash = row['strategy_hash']
                
                if strategy_hash and strategy_hash not in traces:
                    trace = self.load_trace(strategy_hash)
                    if trace is not None:
                        traces[strategy_id] = trace
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to load workspace traces: {e}")
            return {}
    
    def get_strategy_metadata(self, strategy_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get full metadata for a strategy.
        
        Args:
            strategy_hash: Strategy hash
            
        Returns:
            Strategy metadata dict or None
        """
        if self._strategy_index is None:
            return None
        
        matches = self._strategy_index[self._strategy_index['strategy_hash'] == strategy_hash]
        if matches.empty:
            return None
        
        # Convert row to dict and clean up
        metadata = matches.iloc[0].to_dict()
        
        # Remove NaN values
        metadata = {k: v for k, v in metadata.items() if pd.notna(v)}
        
        return metadata
    
    def export_for_notebook(self, strategy_hashes: List[str], output_file: str):
        """
        Export traces for use in notebooks.
        
        Args:
            strategy_hashes: List of strategy hashes to export
            output_file: Output parquet file path
        """
        all_traces = []
        
        for hash_val in strategy_hashes:
            trace = self.load_trace(hash_val)
            if trace is not None:
                # Add strategy hash to trace for identification
                trace['strategy_hash'] = hash_val
                all_traces.append(trace)
        
        if all_traces:
            combined = pd.concat(all_traces, ignore_index=True)
            combined.to_parquet(output_file, engine='pyarrow', index=False)
            logger.info(f"Exported {len(all_traces)} traces to {output_file}")
        else:
            logger.warning("No traces found to export")