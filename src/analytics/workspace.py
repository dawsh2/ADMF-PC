# Core Analytics Workspace
"""
Universal SQL-based interface for ADMF-PC analytics.
Primary entry point for all analytics operations.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from datetime import datetime
import json
import hashlib

from .exceptions import AnalyticsError, WorkspaceNotFoundError, QueryError, SchemaError
from .schema import SCHEMA_SQL, get_schema_version
from .functions import TradingFunctions, register_functions


class AnalyticsWorkspace:
    """Universal interface to ADMF-PC analytics
    
    Provides SQL-first access to trading system analysis with optional
    programmatic helpers for complex operations.
    
    Examples:
        # Connect to workspace
        workspace = AnalyticsWorkspace('20250611_grid_search_SPY')
        
        # SQL is the primary interface
        results = workspace.sql('''
            SELECT strategy_type, AVG(sharpe_ratio) as avg_sharpe
            FROM strategies 
            WHERE sharpe_ratio > 1.0
            GROUP BY strategy_type
            ORDER BY avg_sharpe DESC
        ''')
        
        # Optional programmatic helpers
        signals = workspace.load_signals('momentum_20_30_70_a1b2c3')
    """
    
    def __init__(self, workspace_path: Union[str, Path]):
        """Initialize analytics workspace
        
        Args:
            workspace_path: Path to workspace directory
            
        Raises:
            WorkspaceNotFoundError: If workspace directory doesn't exist
        """
        self.workspace_path = Path(workspace_path)
        
        if not self.workspace_path.exists():
            raise WorkspaceNotFoundError(f"Workspace not found: {workspace_path}")
        
        self.db_path = self.workspace_path / 'analytics.duckdb'
        
        # Auto-initialize if needed
        if not self.db_path.exists():
            self._initialize_workspace()
        
        # Connect to database
        try:
            self.conn = duckdb.connect(str(self.db_path))
            self._setup_workspace_context()
            
            # Initialize trading functions
            self.functions = TradingFunctions(self.workspace_path)
            register_functions(self.conn, self.workspace_path)
        except Exception as e:
            raise AnalyticsError(f"Failed to connect to workspace database: {e}")
    
    def sql(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame
        
        Args:
            query: SQL query string
            params: Optional query parameters for parameterized queries
            
        Returns:
            DataFrame with query results
            
        Raises:
            QueryError: If SQL execution fails
            
        Examples:
            # Simple query
            results = workspace.sql("SELECT COUNT(*) FROM strategies")
            
            # Parameterized query
            results = workspace.sql(
                "SELECT * FROM strategies WHERE sharpe_ratio > ?", 
                [1.5]
            )
        """
        try:
            if params:
                return self.conn.execute(query, params).df()
            else:
                return self.conn.execute(query).df()
        except Exception as e:
            raise QueryError(f"SQL execution failed: {e}\\nQuery: {query}")
    
    def query(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """Alias for sql() method - more intuitive name"""
        return self.sql(query, params)
    
    def describe(self, table: Optional[str] = None) -> pd.DataFrame:
        """Describe tables or specific table structure
        
        Args:
            table: Optional table name. If None, shows all tables.
            
        Returns:
            DataFrame with table structure information
        """
        if table:
            return self.sql(f"DESCRIBE {table}")
        else:
            return self.sql("SHOW TABLES")
    
    def tables(self) -> List[str]:
        """Get list of available tables
        
        Returns:
            List of table names
        """
        result = self.sql("SHOW TABLES")
        return result['name'].tolist() if not result.empty else []
    
    def summary(self) -> Dict[str, Any]:
        """Get workspace summary statistics
        
        Returns:
            Dictionary with workspace summary
        """
        try:
            # Get run information
            runs = self.sql("SELECT COUNT(*) as run_count FROM runs")
            run_count = runs.iloc[0]['run_count'] if not runs.empty else 0
            
            # Get strategy information (pure catalog - no performance metrics)
            strategies = self.sql("""
                SELECT 
                    COUNT(*) as total_strategies,
                    COUNT(DISTINCT strategy_type) as strategy_types
                FROM strategies
            """)
            
            strategy_info = strategies.iloc[0].to_dict() if not strategies.empty else {}
            
            # Get classifier information
            classifiers = self.sql("SELECT COUNT(*) as total_classifiers FROM classifiers")
            classifier_count = classifiers.iloc[0]['total_classifiers'] if not classifiers.empty else 0
            
            return {
                'workspace_path': str(self.workspace_path),
                'database_path': str(self.db_path),
                'run_count': run_count,
                'total_strategies': strategy_info.get('total_strategies', 0),
                'strategy_types': strategy_info.get('strategy_types', 0),
                'total_classifiers': classifier_count,
                # Pure lazy - no pre-computed performance metrics
                'schema_version': get_schema_version()
            }
        except Exception as e:
            raise AnalyticsError(f"Failed to generate summary: {e}")
    
    def load_signals(self, strategy_id_or_file: str) -> pd.DataFrame:
        """Load signal data for a strategy
        
        Args:
            strategy_id_or_file: Strategy ID or direct file path
            
        Returns:
            DataFrame with signal data (sparse format)
            
        Raises:
            AnalyticsError: If strategy not found or file cannot be loaded
        """
        try:
            if strategy_id_or_file.endswith('.parquet'):
                # Direct file path
                file_path = self.workspace_path / strategy_id_or_file
            else:
                # Strategy ID - look up file path
                result = self.sql("""
                    SELECT signal_file_path FROM strategies 
                    WHERE strategy_id = ?
                """, [strategy_id_or_file])
                
                if result.empty:
                    raise AnalyticsError(f"Strategy {strategy_id_or_file} not found")
                
                file_path = self.workspace_path / result.iloc[0]['signal_file_path']
            
            if not file_path.exists():
                raise AnalyticsError(f"Signal file not found: {file_path}")
            
            return pd.read_parquet(file_path)
            
        except Exception as e:
            if isinstance(e, AnalyticsError):
                raise
            raise AnalyticsError(f"Failed to load signals: {e}")
    
    def export_results(self, query: str, output_path: Union[str, Path], 
                      format: str = 'csv', params: Optional[List] = None) -> None:
        """Export query results to file
        
        Args:
            query: SQL query to execute
            output_path: Output file path
            format: Output format ('csv', 'parquet', 'excel')
            params: Optional query parameters
            
        Raises:
            AnalyticsError: If export fails
        """
        try:
            results = self.sql(query, params)
            output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'csv':
                results.to_csv(output_path, index=False)
            elif format.lower() == 'parquet':
                results.to_parquet(output_path, index=False)
            elif format.lower() == 'excel':
                results.to_excel(output_path, index=False)
            else:
                raise AnalyticsError(f"Unsupported export format: {format}")
                
        except Exception as e:
            if isinstance(e, AnalyticsError):
                raise
            raise AnalyticsError(f"Failed to export results: {e}")
    
    # Custom trading function proxies for convenience
    def load_signal_data(self, file_path: str) -> pd.DataFrame:
        """Load signal data using custom function
        
        Args:
            file_path: Relative path to signal file
            
        Returns:
            DataFrame with signal data
        """
        return self.functions.load_signals(file_path)
    
    def load_classifier_states(self, file_path: str) -> pd.DataFrame:
        """Load classifier states using custom function
        
        Args:
            file_path: Relative path to states file
            
        Returns:
            DataFrame with classifier states
        """
        return self.functions.load_states(file_path)
    
    def calculate_signal_correlation(self, file_a: str, file_b: str) -> float:
        """Calculate correlation between two signal files
        
        Args:
            file_a: Path to first signal file
            file_b: Path to second signal file
            
        Returns:
            Correlation coefficient
        """
        return self.functions.signal_correlation(file_a, file_b)
    
    def expand_sparse_signals(self, file_path: str, total_bars: int) -> pd.DataFrame:
        """Expand sparse signals to full timeseries
        
        Args:
            file_path: Path to sparse signal file
            total_bars: Total number of bars
            
        Returns:
            DataFrame with full timeseries
        """
        return self.functions.expand_signals(file_path, total_bars)
    
    def get_signal_statistics(self, file_path: str) -> Dict[str, Any]:
        """Get signal statistics
        
        Args:
            file_path: Path to signal file
            
        Returns:
            Dictionary with signal statistics
        """
        return self.functions.signal_stats(file_path)
    
    def get_regime_statistics(self, file_path: str) -> Dict[str, Any]:
        """Get regime statistics
        
        Args:
            file_path: Path to states file
            
        Returns:
            Dictionary with regime statistics
        """
        return self.functions.regime_stats(file_path)
    
    def calculate_performance(self, strategy_id: str, market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate performance metrics for a strategy on-demand
        
        This is the core lazy calculation method. Load sparse signals and
        calculate full performance metrics only when needed.
        
        Args:
            strategy_id: Strategy ID to calculate performance for
            market_data: Optional market data DataFrame. If not provided,
                        will need to be loaded based on workspace config.
                        
        Returns:
            Dictionary with calculated performance metrics
            
        Example:
            # Calculate performance for specific strategy
            perf = workspace.calculate_performance('momentum_ma_5_20_001')
            print(f"Sharpe Ratio: {perf['sharpe_ratio']}")
            print(f"Max Drawdown: {perf['max_drawdown']}")
        """
        # Get strategy info from catalog
        strategy_info = self.sql(
            "SELECT * FROM strategies WHERE strategy_id = ?", 
            [strategy_id]
        )
        
        if strategy_info.empty:
            raise AnalyticsError(f"Strategy {strategy_id} not found")
        
        # Load sparse signals
        signal_file_path = strategy_info.iloc[0]['signal_file_path']
        signals = self.load_signals(signal_file_path)
        
        # Calculate performance metrics from signals
        # This would integrate with your existing performance calculation logic
        # For now, returning a template structure
        return {
            'strategy_id': strategy_id,
            'signal_file_path': signal_file_path,
            'total_signals': len(signals),
            # These would be calculated from signals + market data
            'sharpe_ratio': None,
            'total_return': None,
            'max_drawdown': None,
            'total_trades': None,
            'win_rate': None,
            'profit_factor': None,
            # ... more metrics as needed
            'calculated_at': datetime.now().isoformat()
        }
    
    def _initialize_workspace(self) -> None:
        """Initialize new workspace database"""
        try:
            # Create database connection
            conn = duckdb.connect(str(self.db_path))
            
            # Execute schema creation
            conn.execute(SCHEMA_SQL)
            
            # Create metadata table for schema version
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _analytics_metadata (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Store schema version
            conn.execute("""
                INSERT OR REPLACE INTO _analytics_metadata (key, value)
                VALUES ('schema_version', ?)
            """, [get_schema_version()])
            
            conn.close()
            
        except Exception as e:
            raise SchemaError(f"Failed to initialize workspace: {e}")
    
    def _setup_workspace_context(self) -> None:
        """Set up workspace-specific SQL context"""
        try:
            # Store workspace root as instance variable for use in custom functions
            self._workspace_root = str(self.workspace_path.absolute())
            
            # Register custom functions
            self._register_custom_functions()
            
        except Exception as e:
            raise AnalyticsError(f"Failed to setup workspace context: {e}")
    
    def _register_custom_functions(self) -> None:
        """Register trading-specific SQL functions"""
        # Custom functions are now handled by the functions module
        # Accessible via self.functions for programmatic use
        pass
    
    def close(self) -> None:
        """Close database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __repr__(self) -> str:
        """String representation"""
        return f"AnalyticsWorkspace('{self.workspace_path}')"