"""
Simple Analytics for ADMF-PC

Minimal wrapper around DuckDB for analyzing trace data.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Union


class TraceAnalysis:
    """
    Simple DuckDB-based trace analysis.
    
    Example:
        ta = TraceAnalysis()  # auto-finds latest results
        df = ta.sql("SELECT * FROM traces WHERE sharpe > 1.5")
        df.plot.scatter('period', 'sharpe')
    """
    
    def __init__(self, path: Optional[Union[str, Path]] = None):
        """
        Initialize with trace data.
        
        Args:
            path: Path to results directory. If None, searches for latest.
        """
        self.con = duckdb.connect(':memory:')
        
        # Find traces
        if path is None:
            path = self._find_latest_results()
        
        self.path = Path(path)
        
        # Load all parquet files with proper schema
        traces_pattern = str(self.path / "traces/**/*.parquet")
        
        # The sparse format has columns: idx (bar index), val (signal value), px (price)
        # We need to add strategy_id from the filename
        self.con.execute(f"""
            CREATE VIEW traces AS 
            SELECT 
                regexp_extract(filename, 'strategy_(\\d+)', 1)::INT as strategy_id,
                idx as bar_idx,
                val as signal_value,
                px as price
            FROM read_parquet('{traces_pattern}', filename=true)
        """)
        
        # Load metadata if available
        metadata_path = self.path / "metadata_enhanced.json"
        if metadata_path.exists():
            self.con.execute(f"""
                CREATE VIEW metadata AS 
                SELECT * FROM read_json('{metadata_path}')
            """)
    
    def _find_latest_results(self) -> Path:
        """Auto-find latest results directory."""
        # Check common locations
        candidates = [
            Path("results/latest"),
            Path("../results/latest"),
            Path("config/*/results/latest"),
            Path("configs/*/results/latest")
        ]
        
        for pattern in candidates:
            if '*' in str(pattern):
                # Glob pattern
                matches = list(Path('.').glob(str(pattern)))
                if matches:
                    # Return most recent
                    return max(matches, key=lambda p: p.stat().st_mtime)
            elif pattern.exists():
                return pattern
        
        # Search current directory tree
        for results_dir in Path('.').rglob('results'):
            if (results_dir / 'latest').exists():
                return results_dir / 'latest'
        
        raise FileNotFoundError("No results directory found. Specify path explicitly.")
    
    def sql(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as pandas DataFrame
        """
        return self.con.execute(query).df()
    
    def __repr__(self):
        try:
            count = self.con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            return f"TraceAnalysis(path='{self.path}', traces={count})"
        except:
            return f"TraceAnalysis(path='{self.path}')"


# Useful query snippets
QUERIES = {
    'top_performers': """
        SELECT * FROM traces 
        ORDER BY sharpe DESC 
        LIMIT 100
    """,
    
    'filter_profitable': """
        SELECT * FROM traces
        WHERE sharpe > 1.5 
        AND max_drawdown < 0.1
        AND num_trades > 100
    """,
    
    'parameter_analysis': """
        SELECT 
            params->>'period' as period,
            params->>'multiplier' as multiplier,
            AVG(sharpe) as avg_sharpe,
            COUNT(*) as count
        FROM traces
        GROUP BY period, multiplier
        HAVING count > 5
        ORDER BY avg_sharpe DESC
    """,
    
    'robustness_check': """
        WITH param_neighbors AS (
            SELECT 
                t1.strategy_id,
                t1.sharpe as center_sharpe,
                AVG(t2.sharpe) as neighbor_avg_sharpe,
                STDDEV(t2.sharpe) as neighbor_std_sharpe
            FROM traces t1
            JOIN traces t2 ON 
                ABS((t1.params->>'period')::int - (t2.params->>'period')::int) <= 5
            WHERE t1.sharpe > 1.5
            GROUP BY t1.strategy_id, t1.sharpe
        )
        SELECT * FROM param_neighbors
        WHERE neighbor_std_sharpe < 0.3
        ORDER BY center_sharpe DESC
    """
}


def quick_analysis(path: Optional[str] = None):
    """
    Quick start for interactive analysis.
    
    Returns TraceAnalysis instance and prints helpful info.
    """
    ta = TraceAnalysis(path)
    
    # Quick stats
    stats = ta.sql("""
        SELECT 
            COUNT(DISTINCT strategy_id) as total_strategies,
            COUNT(*) as total_signals,
            MIN(bar_idx) as first_bar,
            MAX(bar_idx) as last_bar
        FROM traces
    """).iloc[0]
    
    print(f"Loaded {stats['total_strategies']} strategies")
    print(f"Total signals: {stats['total_signals']}")
    print(f"Bar range: {stats['first_bar']} - {stats['last_bar']}")
    print("\nExample queries:")
    print("  ta.sql('SELECT * FROM traces WHERE strategy_id = 0')")
    print("  ta.sql('SELECT strategy_id, COUNT(*) as signals FROM traces GROUP BY strategy_id')")
    
    return ta