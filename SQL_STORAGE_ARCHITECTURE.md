# SQL Storage Architecture for Large-Scale Grid Search

## Overview

For 5000+ strategies, we need a hybrid approach:
- **SQL Database**: Metadata, performance metrics, parameters
- **Parquet Files**: Raw signal/classifier data (sparse storage)
- **DuckDB**: Zero-config, embedded SQL with excellent Parquet integration

## Database Schema

### Core Tables

```sql
-- Optimization runs (workspaces)
CREATE TABLE runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    workflow_type VARCHAR,  -- grid_search, optimization, backtest
    symbols VARCHAR[],
    start_date DATE,
    end_date DATE,
    total_bars INTEGER,
    config_hash VARCHAR,
    status VARCHAR,  -- running, completed, failed
    workspace_path VARCHAR
);

-- Strategy definitions and results
CREATE TABLE strategies (
    strategy_id VARCHAR PRIMARY KEY,
    run_id VARCHAR REFERENCES runs(run_id),
    strategy_type VARCHAR,  -- ma_crossover, momentum, etc.
    parameters JSONB,
    
    -- File locations
    signal_file VARCHAR,
    signal_changes INTEGER,
    compression_ratio REAL,
    
    -- Performance metrics (after execution costs)
    total_return REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    win_rate REAL,
    profit_factor REAL,
    calmar_ratio REAL,
    
    -- Trade statistics
    total_trades INTEGER,
    avg_trade_duration REAL,
    largest_win REAL,
    largest_loss REAL,
    
    -- Execution costs
    gross_sharpe REAL,
    net_sharpe REAL,
    total_commission REAL,
    total_slippage REAL,
    
    -- Timestamps
    created_at TIMESTAMP,
    processed_at TIMESTAMP
);

-- Classifier definitions and results
CREATE TABLE classifiers (
    classifier_id VARCHAR PRIMARY KEY,
    run_id VARCHAR REFERENCES runs(run_id),
    classifier_type VARCHAR,
    parameters JSONB,
    
    -- File locations
    state_file VARCHAR,
    regime_changes INTEGER,
    compression_ratio REAL,
    
    -- Statistics
    regime_counts JSONB,  -- {regime: count}
    avg_regime_duration REAL,
    regime_stability REAL,
    transition_matrix JSONB,
    
    created_at TIMESTAMP
);

-- Strategy-Classifier performance combinations
CREATE TABLE regime_performance (
    id SERIAL PRIMARY KEY,
    strategy_id VARCHAR REFERENCES strategies(strategy_id),
    classifier_id VARCHAR REFERENCES classifiers(classifier_id),
    regime VARCHAR,
    
    -- Regime-specific metrics
    sharpe_ratio REAL,
    total_return REAL,
    win_rate REAL,
    trade_count INTEGER,
    avg_duration REAL,
    
    -- Combined score
    regime_aware_score REAL
);

-- Parameter sensitivity analysis
CREATE TABLE parameter_analysis (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR REFERENCES runs(run_id),
    strategy_type VARCHAR,
    parameter_name VARCHAR,
    parameter_value REAL,
    
    -- Aggregated metrics
    avg_sharpe REAL,
    std_sharpe REAL,
    count_strategies INTEGER,
    best_sharpe REAL,
    
    UNIQUE(run_id, strategy_type, parameter_name, parameter_value)
);

-- Indexes for fast queries
CREATE INDEX idx_strategies_performance ON strategies(sharpe_ratio DESC, max_drawdown);
CREATE INDEX idx_strategies_type_params ON strategies(strategy_type, parameters);
CREATE INDEX idx_strategies_run ON strategies(run_id);
CREATE INDEX idx_regime_perf_combo ON regime_performance(strategy_id, classifier_id);
CREATE INDEX idx_param_analysis ON parameter_analysis(run_id, strategy_type, parameter_name);
```

## DuckDB Implementation

### Why DuckDB?
- **Zero setup**: Embedded, no server needed
- **Excellent Parquet integration**: Can query Parquet files directly
- **Fast analytics**: Columnar storage, vectorized queries
- **Python integration**: Seamless with pandas/pyarrow

```python
# src/analytics/storage/sql_backend.py

import duckdb
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

class DuckDBBackend:
    """DuckDB backend for analytics storage"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        schema_sql = Path(__file__).parent / 'schema.sql'
        with open(schema_sql) as f:
            self.conn.execute(f.read())
    
    def insert_run(self, run_metadata: Dict) -> None:
        """Insert new optimization run"""
        self.conn.execute("""
            INSERT INTO runs (run_id, created_at, workflow_type, symbols, 
                            start_date, end_date, total_bars, config_hash, 
                            status, workspace_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_metadata['run_id'],
            run_metadata['created_at'],
            run_metadata['workflow_type'],
            run_metadata['symbols'],
            run_metadata['start_date'],
            run_metadata['end_date'],
            run_metadata['total_bars'],
            run_metadata['config_hash'],
            'running',
            str(run_metadata['workspace_path'])
        ])
    
    def batch_insert_strategies(self, strategies: List[Dict]) -> None:
        """Efficiently insert many strategies"""
        df = pd.DataFrame(strategies)
        self.conn.register('strategy_batch', df)
        
        self.conn.execute("""
            INSERT INTO strategies 
            SELECT * FROM strategy_batch
        """)
        
        self.conn.unregister('strategy_batch')
    
    def scan_performance(
        self, 
        min_sharpe: Optional[float] = None,
        strategy_types: Optional[List[str]] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fast performance scanning with SQL"""
        
        conditions = ["1=1"]  # Always true base condition
        params = []
        
        if min_sharpe:
            conditions.append("sharpe_ratio >= ?")
            params.append(min_sharpe)
        
        if strategy_types:
            placeholders = ','.join(['?' for _ in strategy_types])
            conditions.append(f"strategy_type IN ({placeholders})")
            params.extend(strategy_types)
        
        query = f"""
            SELECT 
                s.strategy_id,
                s.strategy_type,
                s.parameters,
                s.sharpe_ratio,
                s.max_drawdown,
                s.win_rate,
                s.total_trades,
                s.signal_changes,
                r.run_id,
                r.created_at
            FROM strategies s
            JOIN runs r ON s.run_id = r.run_id
            WHERE {' AND '.join(conditions)}
            ORDER BY s.sharpe_ratio DESC
            LIMIT ?
        """
        
        params.append(limit)
        return self.conn.execute(query, params).df()
    
    def parameter_sensitivity(
        self, 
        run_id: str, 
        strategy_type: str, 
        parameter: str
    ) -> pd.DataFrame:
        """Analyze parameter sensitivity"""
        
        # First check if pre-computed
        result = self.conn.execute("""
            SELECT parameter_value, avg_sharpe, std_sharpe, count_strategies
            FROM parameter_analysis
            WHERE run_id = ? AND strategy_type = ? AND parameter_name = ?
            ORDER BY parameter_value
        """, [run_id, strategy_type, parameter]).df()
        
        if not result.empty:
            return result
        
        # Compute on-the-fly if not cached
        return self.conn.execute("""
            SELECT 
                CAST(parameters->>? AS REAL) as parameter_value,
                AVG(sharpe_ratio) as avg_sharpe,
                STDDEV(sharpe_ratio) as std_sharpe,
                COUNT(*) as count_strategies,
                MAX(sharpe_ratio) as best_sharpe
            FROM strategies
            WHERE run_id = ? 
                AND strategy_type = ?
                AND parameters->>? IS NOT NULL
            GROUP BY CAST(parameters->>? AS REAL)
            ORDER BY parameter_value
        """, [parameter, run_id, strategy_type, parameter, parameter]).df()
    
    def top_combinations(
        self, 
        run_id: str, 
        metric: str = 'regime_aware_score',
        limit: int = 20
    ) -> pd.DataFrame:
        """Find best strategy-classifier combinations"""
        
        return self.conn.execute(f"""
            SELECT 
                rp.strategy_id,
                rp.classifier_id,
                s.strategy_type,
                c.classifier_type,
                s.parameters as strategy_params,
                c.parameters as classifier_params,
                AVG(rp.{metric}) as avg_score,
                COUNT(rp.regime) as regime_count
            FROM regime_performance rp
            JOIN strategies s ON rp.strategy_id = s.strategy_id
            JOIN classifiers c ON rp.classifier_id = c.classifier_id
            WHERE s.run_id = ?
            GROUP BY rp.strategy_id, rp.classifier_id, 
                     s.strategy_type, c.classifier_type,
                     s.parameters, c.parameters
            ORDER BY avg_score DESC
            LIMIT ?
        """, [run_id, limit]).df()
    
    def strategy_correlations(self, run_id: str) -> pd.DataFrame:
        """Calculate strategy correlations from signals"""
        
        # This would load signal data and compute correlations
        # Could be pre-computed and cached
        query = """
            WITH signal_data AS (
                SELECT 
                    strategy_id,
                    -- Load and correlate signals from Parquet files
                    corr_with_signals(signal_file) as correlations
                FROM strategies
                WHERE run_id = ?
            )
            SELECT * FROM signal_data
        """
        
        # For now, return placeholder
        return pd.DataFrame()
```

## Hybrid Storage Pattern

```python
# Storage pattern for 5000+ strategies
class HybridStorageManager:
    """Combines SQL metadata with Parquet signal storage"""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.db = DuckDBBackend(workspace_path / 'analytics.duckdb')
        self.signal_storage = ParquetSignalStorage(workspace_path / 'signals')
    
    def store_grid_search_results(
        self, 
        run_metadata: Dict,
        strategies: List[Dict],
        classifiers: List[Dict]
    ):
        """Efficiently store large grid search results"""
        
        # 1. Insert run metadata
        self.db.insert_run(run_metadata)
        
        # 2. Batch insert strategies (much faster than individual inserts)
        self.db.batch_insert_strategies(strategies)
        
        # 3. Store signals in partitioned Parquet files
        self.signal_storage.store_partitioned_signals(strategies)
        
        # 4. Pre-compute analytics
        self.pre_compute_analytics(run_metadata['run_id'])
    
    def pre_compute_analytics(self, run_id: str):
        """Pre-compute expensive analytics"""
        
        # Parameter sensitivity for all strategy types
        strategy_types = self.db.conn.execute("""
            SELECT DISTINCT strategy_type FROM strategies WHERE run_id = ?
        """, [run_id]).fetchall()
        
        for (strategy_type,) in strategy_types:
            # Get all parameters for this strategy type
            params = self.db.conn.execute("""
                SELECT DISTINCT jsonb_object_keys(parameters) as param_name
                FROM strategies 
                WHERE run_id = ? AND strategy_type = ?
            """, [run_id, strategy_type]).fetchall()
            
            for (param_name,) in params:
                # Compute and cache parameter sensitivity
                sensitivity = self.db.parameter_sensitivity(run_id, strategy_type, param_name)
                self._cache_parameter_analysis(run_id, strategy_type, param_name, sensitivity)
```

## CLI Integration

```bash
# Fast SQL-powered queries
admf analytics scan --min-sharpe 1.5 --top 50
# SELECT * FROM strategies WHERE sharpe_ratio >= 1.5 ORDER BY sharpe_ratio DESC LIMIT 50

admf analytics analyze-parameters fc4bb91c --strategy momentum --parameter sma_period
# Loads pre-computed parameter sensitivity from SQL

admf analytics correlations fc4bb91c --min-correlation 0.7
# Fast correlation queries from cached results

# Complex filtering
admf analytics scan \
    --strategy-types momentum,ma_crossover \
    --min-sharpe 1.2 \
    --max-drawdown 0.15 \
    --min-trades 100
# Efficient WHERE clause with indexes
```

## Performance Benefits

### File-Based (Current)
- **5000 strategies**: Load 5000 JSON files or 1 huge index
- **Query time**: O(n) linear scan through all records
- **Memory**: Must load all metadata to filter
- **Aggregations**: Require loading all data

### SQL-Based (Proposed)
- **5000 strategies**: Single query with indexes
- **Query time**: O(log n) with proper indexes
- **Memory**: Only query results loaded
- **Aggregations**: Native SQL GROUP BY, AVG, etc.

## Migration Path

1. **Phase 1**: Add SQL backend alongside file storage
2. **Phase 2**: Migrate existing workspaces to SQL
3. **Phase 3**: Use SQL as primary, keep Parquet for signals
4. **Phase 4**: Add advanced analytics (correlations, regime analysis)

## Specific Improvements for 5000+ Strategies

1. **Batched Inserts**: Insert 1000s of strategies in single transaction
2. **Partitioned Storage**: Split signals by strategy type
3. **Indexed Queries**: Fast filtering on performance metrics
4. **Pre-computed Analytics**: Cache expensive calculations
5. **Streaming Results**: Don't load all 5000 results at once

Would you like me to implement the SQL backend? It would make the system much more scalable and enable sophisticated analytics queries.