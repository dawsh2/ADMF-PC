# Universal Analytics Interface Design
## SQL-First Library with Optional Programmatic Access

## Core Philosophy

**SQL as the Universal Interface**
- Primary interaction through standard SQL
- Library handles workspace setup and connection
- Optional Python helpers for complex operations
- No custom CLI commands to learn - just SQL

## Architecture

```python
# Simple entry point
from admf_analytics import AnalyticsWorkspace

# Connect to workspace
workspace = AnalyticsWorkspace('20250611_grid_search_SPY')

# SQL is the primary interface
results = workspace.sql("""
    SELECT strategy_type, AVG(sharpe_ratio) as avg_sharpe
    FROM strategies 
    WHERE sharpe_ratio > 1.0
    GROUP BY strategy_type
    ORDER BY avg_sharpe DESC
""")

# Optional: programmatic helpers for complex operations
signals = workspace.load_signals('momentum_20_30_70_a1b2c3')
correlations = workspace.compute_correlations(['strategy_1', 'strategy_2'])
```

## Core Library Design

### 1. **Single Entry Point**

```python
# src/analytics/workspace.py

import duckdb
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

class AnalyticsWorkspace:
    """Universal interface to ADMF-PC analytics"""
    
    def __init__(self, workspace_path: Union[str, Path]):
        self.workspace_path = Path(workspace_path)
        self.db_path = self.workspace_path / 'analytics.duckdb'
        
        # Auto-initialize if needed
        if not self.db_path.exists():
            self._initialize_workspace()
        
        # Connect to database
        self.conn = duckdb.connect(str(self.db_path))
        self._setup_workspace_context()
    
    def sql(self, query: str, params: List = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            DataFrame with results
        """
        try:
            if params:
                return self.conn.execute(query, params).df()
            else:
                return self.conn.execute(query).df()
        except Exception as e:
            raise AnalyticsError(f"SQL Error: {e}")
    
    def query(self, query: str, params: List = None) -> pd.DataFrame:
        """Alias for sql() - more intuitive name"""
        return self.sql(query, params)
    
    def describe(self, table: str = None) -> pd.DataFrame:
        """Describe tables or specific table structure"""
        if table:
            return self.sql(f"DESCRIBE {table}")
        else:
            return self.sql("SHOW TABLES")
    
    def _setup_workspace_context(self):
        """Set up workspace-specific SQL context"""
        # Make workspace path available to SQL
        self.conn.execute(f"SET workspace_root = '{self.workspace_path.absolute()}'")
        
        # Register custom functions
        self._register_custom_functions()
    
    def _register_custom_functions(self):
        """Register trading-specific SQL functions"""
        
        # Function to load signals from file
        @self.conn.function
        def load_signals(file_path: str) -> pd.DataFrame:
            full_path = self.workspace_path / file_path
            return pd.read_parquet(full_path)
        
        # Function to calculate signal correlation
        @self.conn.function  
        def signal_correlation(file_a: str, file_b: str) -> float:
            signals_a = pd.read_parquet(self.workspace_path / file_a)
            signals_b = pd.read_parquet(self.workspace_path / file_b)
            # Align and correlate signals
            return self._correlate_sparse_signals(signals_a, signals_b)
```

### 2. **SQL Extensions for Trading**

```sql
-- Custom SQL functions available in workspace context

-- Load signal data directly in SQL
SELECT * FROM load_signals('signals/momentum/mom_20_30_a1b2c3.parquet')
WHERE signal != 0;

-- Calculate correlations in SQL
SELECT signal_correlation(
    'signals/momentum/mom_20_30_a1b2c3.parquet',
    'signals/momentum/mom_10_20_b2c3d4.parquet'
) as correlation;

-- Expand sparse signals to full timeseries
SELECT * FROM expand_signals('signals/momentum/mom_20_30_a1b2c3.parquet', 19500);

-- Load and analyze events
SELECT event_type, COUNT(*) 
FROM load_events('events/run_abc/strategy_events.parquet')
GROUP BY event_type;
```

### 3. **Minimal Programmatic Helpers**

```python
# Only for operations that are genuinely complex in SQL
class AnalyticsWorkspace:
    
    def load_signals(self, strategy_id_or_file: str) -> pd.DataFrame:
        """Load signal data for a strategy"""
        if strategy_id_or_file.endswith('.parquet'):
            # Direct file path
            return pd.read_parquet(self.workspace_path / strategy_id_or_file)
        else:
            # Strategy ID - look up file path
            result = self.sql("""
                SELECT signal_file_path FROM strategies 
                WHERE strategy_id = ?
            """, [strategy_id_or_file])
            
            if result.empty:
                raise ValueError(f"Strategy {strategy_id_or_file} not found")
            
            file_path = result.iloc[0]['signal_file_path']
            return pd.read_parquet(self.workspace_path / file_path)
    
    def compare_strategies(self, strategy_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """Load and compare multiple strategies"""
        strategies = {}
        
        for strategy_id in strategy_ids:
            strategies[strategy_id] = self.load_signals(strategy_id)
        
        return strategies
    
    def export_results(self, query: str, output_path: str, format: str = 'csv'):
        """Export query results to file"""
        results = self.sql(query)
        
        if format == 'csv':
            results.to_csv(output_path, index=False)
        elif format == 'parquet':
            results.to_parquet(output_path, index=False)
        elif format == 'excel':
            results.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
```

## Usage Patterns

### 1. **Pure SQL Exploration**

```python
# Connect and explore
workspace = AnalyticsWorkspace('my_grid_search_run')

# Standard SQL - no learning curve
workspace.sql("SELECT COUNT(*) FROM strategies")

workspace.sql("""
    SELECT strategy_type, 
           AVG(sharpe_ratio) as avg_sharpe,
           COUNT(*) as count
    FROM strategies 
    WHERE total_trades > 50
    GROUP BY strategy_type
    ORDER BY avg_sharpe DESC
""")

# Parameter analysis
workspace.sql("""
    SELECT 
        CAST(parameters->>'sma_period' AS INT) as sma_period,
        AVG(sharpe_ratio) as avg_sharpe,
        COUNT(*) as count
    FROM strategies 
    WHERE strategy_type = 'momentum'
    GROUP BY CAST(parameters->>'sma_period' AS INT)
    ORDER BY sma_period
""")
```

### 2. **SQL + Light Programmatic**

```python
# Find top strategies with SQL
top_strategies = workspace.sql("""
    SELECT strategy_id, signal_file_path, sharpe_ratio
    FROM strategies 
    WHERE sharpe_ratio > 1.5
    ORDER BY sharpe_ratio DESC
    LIMIT 10
""")

# Load actual signals for detailed analysis
for _, row in top_strategies.iterrows():
    signals = workspace.load_signals(row['strategy_id'])
    print(f"{row['strategy_id']}: {len(signals)} signals")
```

### 3. **Interactive Session**

```python
# Start interactive session
workspace = AnalyticsWorkspace('my_run')

# Direct DuckDB access for power users
workspace.conn.execute("SELECT * FROM strategies LIMIT 5").show()

# Or use workspace wrapper
workspace.describe()  # Show all tables
workspace.describe('strategies')  # Show strategy table structure
```

## Universal Interface Benefits

### 1. **No Learning Curve**
- Everyone knows SQL
- No custom commands to memorize
- Standard database patterns

### 2. **Powerful and Flexible**
- Full SQL capabilities (joins, window functions, CTEs)
- Custom functions for trading-specific operations
- Direct Parquet file access

### 3. **Tool Agnostic**
- Use with Jupyter notebooks
- Command-line DuckDB client
- Any SQL client that supports DuckDB
- Python scripts

### 4. **Composable**
- Save common queries as views
- Build complex analysis step by step
- Export results for further processing

## Integration Examples

### 1. **Jupyter Notebook**

```python
# analytics_exploration.ipynb
import sys
sys.path.append('/path/to/admf-pc/src')

from analytics.workspace import AnalyticsWorkspace
import plotly.express as px

# Connect
ws = AnalyticsWorkspace('workspaces/20250611_grid_search_SPY')

# Analyze parameter sensitivity
param_data = ws.sql("""
    SELECT 
        CAST(parameters->>'sma_period' AS INT) as sma_period,
        AVG(sharpe_ratio) as avg_sharpe,
        STDDEV(sharpe_ratio) as sharpe_std,
        COUNT(*) as count
    FROM strategies 
    WHERE strategy_type = 'momentum'
    GROUP BY CAST(parameters->>'sma_period' AS INT)
    ORDER BY sma_period
""")

# Visualize
px.line(param_data, x='sma_period', y='avg_sharpe', 
        error_y='sharpe_std', title='SMA Period Sensitivity')
```

### 2. **DuckDB CLI Direct**

```bash
# Connect directly to any workspace
duckdb workspaces/20250611_grid_search_SPY/analytics.duckdb

# Standard SQL exploration
D SELECT * FROM strategies WHERE sharpe_ratio > 2.0;
D SELECT strategy_type, COUNT(*) FROM strategies GROUP BY strategy_type;

# Custom functions available
D SELECT signal_correlation('signals/momentum/mom_20_30.parquet', 
                           'signals/momentum/mom_10_20.parquet');
```

### 3. **Python Scripts**

```python
#!/usr/bin/env python
# daily_analysis.py

from analytics.workspace import AnalyticsWorkspace

def analyze_latest_run():
    # Find latest workspace
    latest = find_latest_workspace()
    ws = AnalyticsWorkspace(latest)
    
    # Standard analysis
    summary = ws.sql("""
        SELECT 
            strategy_type,
            COUNT(*) as total_strategies,
            AVG(sharpe_ratio) as avg_sharpe,
            MAX(sharpe_ratio) as best_sharpe,
            COUNT(*) FILTER (WHERE sharpe_ratio > 1.5) as high_performers
        FROM strategies
        GROUP BY strategy_type
        ORDER BY avg_sharpe DESC
    """)
    
    print("Strategy Performance Summary:")
    print(summary.to_string(index=False))
    
    # Export for reporting
    ws.export_results(
        "SELECT * FROM strategies WHERE sharpe_ratio > 1.5",
        "top_performers.csv"
    )

if __name__ == '__main__':
    analyze_latest_run()
```

## Implementation Structure

```
src/analytics/
├── __init__.py              # Export AnalyticsWorkspace
├── workspace.py             # Main workspace class
├── functions.py             # Custom SQL functions
├── migration.py             # Workspace migration utilities
├── schema.py                # SQL schema definitions
└── utils.py                 # Helper utilities

# Simple import
from analytics import AnalyticsWorkspace
```

## Setup and Migration

```python
# One-time workspace setup
from analytics import setup_workspace, migrate_workspace

# Create new SQL workspace from existing data
setup_workspace('workspaces/my_new_run')

# Migrate existing JSON workspace
migrate_workspace(
    source='workspaces/fc4bb91c-old-format',
    destination='workspaces/fc4bb91c-sql-format'
)

# Then use normally
workspace = AnalyticsWorkspace('workspaces/fc4bb91c-sql-format')
```

This approach gives you:

1. **Universal SQL interface** - works with any SQL tool
2. **Zero learning curve** - standard SQL syntax
3. **Maximum flexibility** - full database capabilities
4. **Optional helpers** - for genuinely complex operations
5. **Tool independence** - use with Jupyter, CLI, scripts, or any SQL client

The SQL becomes your universal API, whether you're in a Jupyter notebook, command line, or Python script. Much cleaner than building yet another CLI!
