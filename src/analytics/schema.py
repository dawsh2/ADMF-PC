# SQL Schema Definitions
"""
Database schema for ADMF-PC analytics storage.
Based on the comprehensive SQL architecture design.
"""

# Core schema SQL
SCHEMA_SQL = """
-- =============================================
-- WORKSPACE AND RUN MANAGEMENT
-- =============================================

CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    workflow_type VARCHAR,              -- 'grid_search', 'optimization', 'backtest'
    
    -- Data characteristics
    symbols VARCHAR[],
    timeframes VARCHAR[],
    start_date DATE,
    end_date DATE,
    total_bars INTEGER,
    
    -- Configuration
    config_file VARCHAR,
    config_hash VARCHAR,
    
    -- Execution details
    total_strategies INTEGER,
    total_classifiers INTEGER,
    total_combinations INTEGER,         -- strategies × classifiers
    
    -- Status and performance
    status VARCHAR,                     -- 'running', 'completed', 'failed'
    duration_seconds REAL,
    peak_memory_mb REAL,
    
    -- Storage details
    workspace_path VARCHAR,
    total_size_mb REAL,
    compression_ratio REAL
);

-- =============================================
-- STRATEGY CATALOG AND PERFORMANCE
-- =============================================

CREATE TABLE IF NOT EXISTS strategies (
    strategy_id VARCHAR PRIMARY KEY,
    run_id VARCHAR REFERENCES runs(run_id),
    
    -- Strategy definition
    strategy_type VARCHAR,              -- 'momentum', 'ma_crossover', 'mean_reversion'
    strategy_name VARCHAR,
    parameters JSON,                    -- {sma_period: 20, rsi_threshold: 30, ...}
    
    -- File references (ONLY thing we store - everything else derived)
    signal_file_path VARCHAR,           -- 'signals/momentum/mom_20_30_a1b2c3.parquet'
    config_hash VARCHAR,                -- Hash of parameters for deduplication
    
    -- Timestamps
    created_at TIMESTAMP
);

-- =============================================
-- CLASSIFIER CATALOG
-- =============================================

CREATE TABLE IF NOT EXISTS classifiers (
    classifier_id VARCHAR PRIMARY KEY,
    run_id VARCHAR REFERENCES runs(run_id),
    
    -- Classifier definition
    classifier_type VARCHAR,            -- 'momentum_regime', 'volatility', 'trend'
    classifier_name VARCHAR,
    parameters JSON,
    
    -- File references (ONLY thing we store - everything else derived)
    states_file_path VARCHAR,           -- 'classifiers/regime/hmm_3state_f6g7h8.parquet'
    config_hash VARCHAR,
    
    -- Timestamps
    created_at TIMESTAMP
);

-- =============================================
-- PURE LAZY DESIGN - NO PRE-COMPUTED TABLES
-- Everything is derived on-demand from sparse signal files
-- =============================================

-- =============================================
-- EVENT ARCHIVE CATALOG
-- =============================================

CREATE TABLE IF NOT EXISTS event_archives (
    archive_id VARCHAR PRIMARY KEY,
    run_id VARCHAR REFERENCES runs(run_id),
    container_id VARCHAR,
    container_type VARCHAR,             -- 'strategy', 'portfolio', 'execution'
    
    -- File references
    events_file_path VARCHAR,           -- 'events/run_abc/strategy_events_part_001.parquet'
    
    -- Event characteristics
    event_types VARCHAR,                -- JSON array as string: ['SIGNAL', 'ORDER', 'FILL']
    event_count_by_type JSON,           -- {'SIGNAL': 1250, 'ORDER': 400, 'FILL': 380}
    total_events INTEGER,
    
    -- Time range
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds REAL,
    
    -- Storage details
    file_size_mb REAL,
    compression_ratio REAL,
    
    created_at TIMESTAMP
);

-- NO PRE-COMPUTED ANALYTICS TABLES
-- All analytics are calculated on-demand from sparse signal files

-- =============================================
-- MINIMAL INDEXES FOR PURE CATALOG
-- =============================================

-- Strategy lookup indexes
CREATE INDEX IF NOT EXISTS idx_strategies_type 
    ON strategies(strategy_type);
CREATE INDEX IF NOT EXISTS idx_strategies_run 
    ON strategies(run_id);
CREATE INDEX IF NOT EXISTS idx_strategies_created 
    ON strategies(created_at DESC);

-- Classifier lookup indexes
CREATE INDEX IF NOT EXISTS idx_classifiers_type 
    ON classifiers(classifier_type);
CREATE INDEX IF NOT EXISTS idx_classifiers_run 
    ON classifiers(run_id);

-- Event archive indexes
CREATE INDEX IF NOT EXISTS idx_events_run_type 
    ON event_archives(run_id, container_type);
CREATE INDEX IF NOT EXISTS idx_events_time_range 
    ON event_archives(start_time, end_time);
"""

# Helper functions for schema management
def get_schema_version() -> str:
    """Get current schema version"""
    return "1.0.0"

def get_migration_scripts() -> dict:
    """Get migration scripts for schema updates"""
    return {
        "1.0.0": SCHEMA_SQL
    }