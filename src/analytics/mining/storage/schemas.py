"""
SQL schema definitions for ADMF-PC analytics database.
Implements the structured metrics layer of the two-layer architecture.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import json
from enum import Enum


class MarketRegime(str, Enum):
    """Market regime classifications."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE_BOUND = "RANGE_BOUND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    REGIME_TRANSITION = "REGIME_TRANSITION"


class VolatilityRegime(str, Enum):
    """Volatility regime classifications."""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


# SQL Schema Creation Scripts
OPTIMIZATION_RUNS_SCHEMA = """
-- Main optimization runs table
CREATE TABLE IF NOT EXISTS optimization_runs (
    run_id UUID PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    correlation_id VARCHAR(100) UNIQUE NOT NULL,
    
    -- Strategy information
    strategy_type VARCHAR(50) NOT NULL,
    strategy_version VARCHAR(20),
    parameters JSONB NOT NULL,
    
    -- Performance metrics
    total_return DECIMAL(10,3),
    sharpe_ratio DECIMAL(5,3),
    sortino_ratio DECIMAL(5,3),
    max_drawdown DECIMAL(5,3),
    win_rate DECIMAL(5,3),
    profit_factor DECIMAL(5,2),
    
    -- Risk metrics
    value_at_risk DECIMAL(10,3),
    expected_shortfall DECIMAL(10,3),
    beta DECIMAL(5,3),
    
    -- Market conditions during run
    market_regime VARCHAR(20),
    volatility_regime VARCHAR(20),
    correlation_regime VARCHAR(20),
    avg_market_volatility DECIMAL(6,4),
    
    -- Execution statistics
    total_trades INTEGER,
    avg_trade_duration INTERVAL,
    avg_slippage DECIMAL(8,4),
    total_commission DECIMAL(10,2),
    
    -- Event stream metadata
    event_count INTEGER,
    first_event_id VARCHAR(100),
    last_event_id VARCHAR(100),
    event_storage_path VARCHAR(255)
);

-- Indexes for fast queries
CREATE INDEX idx_high_sharpe_volatility 
ON optimization_runs(sharpe_ratio, volatility_regime) 
WHERE sharpe_ratio > 1.5;

CREATE INDEX idx_regime_performance
ON optimization_runs(market_regime, sharpe_ratio);

CREATE INDEX idx_strategy_params
ON optimization_runs USING GIN (parameters);

CREATE INDEX idx_correlation_lookup
ON optimization_runs(correlation_id);

CREATE INDEX idx_created_at
ON optimization_runs(created_at);
"""

TRADES_SCHEMA = """
-- Detailed trade analysis
CREATE TABLE IF NOT EXISTS trades (
    trade_id UUID PRIMARY KEY,
    run_id UUID REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    correlation_id VARCHAR(100) NOT NULL,
    
    -- Trade details
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    symbol VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    
    -- Prices and sizes
    entry_price DECIMAL(10,2) NOT NULL,
    exit_price DECIMAL(10,2),
    position_size INTEGER NOT NULL,
    
    -- Performance
    pnl DECIMAL(10,2),
    pnl_percent DECIMAL(6,3),
    
    -- Market context
    entry_volatility DECIMAL(6,4),
    exit_volatility DECIMAL(6,4),
    entry_regime VARCHAR(20),
    exit_regime VARCHAR(20),
    
    -- Event linkage
    entry_signal_event_id VARCHAR(100),
    exit_signal_event_id VARCHAR(100),
    
    -- Metadata
    slippage DECIMAL(8,4),
    commission DECIMAL(8,2),
    
    -- Constraints
    CONSTRAINT valid_direction CHECK (direction IN ('LONG', 'SHORT')),
    CONSTRAINT valid_times CHECK (exit_time IS NULL OR exit_time > entry_time)
);

CREATE INDEX idx_trade_run ON trades(run_id);
CREATE INDEX idx_trade_symbol ON trades(symbol);
CREATE INDEX idx_trade_time ON trades(entry_time, exit_time);
CREATE INDEX idx_trade_pnl ON trades(pnl_percent);
"""

DISCOVERED_PATTERNS_SCHEMA = """
-- Pattern Library (Living Knowledge Base)
CREATE TABLE IF NOT EXISTS discovered_patterns (
    pattern_id UUID PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,  -- 'entry', 'exit', 'risk', 'regime'
    pattern_name VARCHAR(200) NOT NULL,
    pattern_signature JSONB NOT NULL,   -- The actual pattern definition
    
    -- Performance metrics
    success_rate DECIMAL(5,3) NOT NULL,
    sample_count INTEGER NOT NULL,
    confidence_interval DECIMAL(5,3),
    
    -- Validation
    last_validated TIMESTAMP,
    validation_method VARCHAR(100),
    out_of_sample_performance DECIMAL(5,3),
    
    -- Context
    market_conditions JSONB,   -- When pattern works best
    anti_pattern BOOLEAN DEFAULT FALSE,  -- Patterns to avoid
    discovery_method VARCHAR(100), -- How we found it
    correlation_ids TEXT[],    -- Example instances
    
    -- Performance tracking
    live_performance DECIMAL(5,3),  -- How it's doing in production
    backtest_performance DECIMAL(5,3),
    degradation_rate DECIMAL(5,3),   -- Performance decay over time
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',  -- active, degraded, retired
    
    CONSTRAINT valid_pattern_type CHECK (
        pattern_type IN ('entry', 'exit', 'risk', 'regime', 'composite')
    ),
    CONSTRAINT valid_status CHECK (
        status IN ('active', 'degraded', 'retired', 'testing')
    )
);

CREATE INDEX idx_pattern_signature 
ON discovered_patterns USING GIN (pattern_signature);

CREATE INDEX idx_pattern_performance
ON discovered_patterns(success_rate, pattern_type)
WHERE anti_pattern = false AND status = 'active';

CREATE INDEX idx_pattern_type_status
ON discovered_patterns(pattern_type, status);
"""

PATTERN_PERFORMANCE_HISTORY_SCHEMA = """
-- Track pattern performance over time
CREATE TABLE IF NOT EXISTS pattern_performance_history (
    pattern_id UUID REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE,
    evaluation_date DATE NOT NULL,
    
    -- Performance metrics
    success_rate DECIMAL(5,3) NOT NULL,
    sample_size INTEGER NOT NULL,
    avg_return DECIMAL(10,3),
    sharpe_contribution DECIMAL(5,3),
    
    -- Error metrics
    false_positive_rate DECIMAL(5,3),
    false_negative_rate DECIMAL(5,3),
    
    -- Market context
    market_conditions JSONB,
    regime_distribution JSONB,  -- {regime: percentage}
    
    PRIMARY KEY (pattern_id, evaluation_date)
);

CREATE INDEX idx_pattern_history_date 
ON pattern_performance_history(evaluation_date);
"""

PATTERN_INTERACTIONS_SCHEMA = """
-- Track pattern co-occurrences and their outcomes
CREATE TABLE IF NOT EXISTS pattern_interactions (
    interaction_id UUID PRIMARY KEY,
    pattern_a_id UUID REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE,
    pattern_b_id UUID REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE,
    
    -- Occurrence metrics
    co_occurrence_count INTEGER NOT NULL,
    last_observed TIMESTAMP,
    
    -- Performance metrics when both active
    joint_success_rate DECIMAL(5,3),
    joint_sharpe DECIMAL(5,3),
    joint_return DECIMAL(10,3),
    
    -- Interaction metrics
    synergy_score DECIMAL(5,3),  -- Positive = amplifying, Negative = canceling
    correlation DECIMAL(4,3),
    interaction_type VARCHAR(20),  -- 'synergistic', 'neutral', 'conflicting'
    
    -- Temporal relationships
    avg_time_lag INTERVAL,  -- Does one typically precede the other?
    sequence_matters BOOLEAN,
    preferred_order VARCHAR(10),  -- 'a_first', 'b_first', 'simultaneous'
    
    UNIQUE(pattern_a_id, pattern_b_id),
    CONSTRAINT different_patterns CHECK (pattern_a_id != pattern_b_id),
    CONSTRAINT ordered_patterns CHECK (pattern_a_id < pattern_b_id)  -- Prevent duplicates
);

CREATE INDEX idx_interaction_synergy
ON pattern_interactions(synergy_score)
WHERE co_occurrence_count > 10;
"""

PATTERN_REGIME_PERFORMANCE_SCHEMA = """
-- Store regime-specific pattern performance
CREATE TABLE IF NOT EXISTS pattern_regime_performance (
    pattern_id UUID REFERENCES discovered_patterns(pattern_id) ON DELETE CASCADE,
    regime_type VARCHAR(50) NOT NULL,
    
    -- Regime-specific metrics
    success_rate DECIMAL(5,3) NOT NULL,
    avg_return DECIMAL(10,3),
    sharpe_ratio DECIMAL(5,3),
    max_drawdown DECIMAL(5,3),
    
    -- Optimal parameters for this regime
    optimal_threshold DECIMAL(4,3),
    optimal_parameters JSONB,
    
    -- Sample statistics
    sample_size INTEGER NOT NULL,
    confidence_level DECIMAL(4,3),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (pattern_id, regime_type)
);

CREATE INDEX idx_regime_performance
ON pattern_regime_performance(regime_type, success_rate);
"""

MARKET_CONDITIONS_SCHEMA = """
-- Market conditions during optimization
CREATE TABLE IF NOT EXISTS market_conditions (
    condition_id UUID PRIMARY KEY,
    run_id UUID REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    
    -- Market state
    vix_level DECIMAL(6,2),
    market_regime VARCHAR(20),
    sector_rotation_score DECIMAL(5,3),
    
    -- Correlations
    equity_bond_correlation DECIMAL(4,3),
    sector_dispersion DECIMAL(6,3),
    correlation_breakdown JSONB,  -- Detailed correlation matrix
    
    -- Microstructure
    avg_spread DECIMAL(8,4),
    avg_volume BIGINT,
    liquidity_score DECIMAL(5,3),
    market_impact_cost DECIMAL(8,4),
    
    -- Additional indicators
    put_call_ratio DECIMAL(5,3),
    advance_decline_ratio DECIMAL(5,3),
    high_low_ratio DECIMAL(5,3)
);

CREATE INDEX idx_market_conditions_run
ON market_conditions(run_id, timestamp);

CREATE INDEX idx_market_regime_time
ON market_conditions(market_regime, timestamp);
"""

PERFORMANCE_ATTRIBUTION_SCHEMA = """
-- Performance attribution analysis
CREATE TABLE IF NOT EXISTS performance_attribution (
    attribution_id UUID PRIMARY KEY,
    run_id UUID REFERENCES optimization_runs(run_id) ON DELETE CASCADE,
    
    -- Attribution components (should sum to total return)
    market_timing DECIMAL(8,3),
    stock_selection DECIMAL(8,3),
    regime_selection DECIMAL(8,3),
    risk_management DECIMAL(8,3),
    execution_cost DECIMAL(8,3),
    
    -- Factor exposures
    momentum_exposure DECIMAL(5,3),
    value_exposure DECIMAL(5,3),
    size_exposure DECIMAL(5,3),
    volatility_exposure DECIMAL(5,3),
    quality_exposure DECIMAL(5,3),
    
    -- Risk attribution
    systematic_risk DECIMAL(5,3),
    idiosyncratic_risk DECIMAL(5,3),
    regime_risk DECIMAL(5,3),
    
    -- Metadata
    attribution_method VARCHAR(50),
    calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_attribution_run
ON performance_attribution(run_id);
"""

EVENT_COMPRESSION_STATS_SCHEMA = """
-- Track compression effectiveness
CREATE TABLE IF NOT EXISTS event_compression_stats (
    compression_id UUID PRIMARY KEY,
    date_range daterange NOT NULL,
    
    -- Compression metrics
    original_event_count BIGINT NOT NULL,
    compressed_event_count BIGINT NOT NULL,
    compression_ratio DECIMAL(5,2) NOT NULL,
    
    -- Storage metrics
    original_size_mb DECIMAL(10,2),
    compressed_size_mb DECIMAL(10,2),
    storage_savings_percent DECIMAL(5,2),
    
    -- Quality metrics
    information_loss_score DECIMAL(4,3),  -- 0 = no loss, 1 = total loss
    pattern_coverage DECIMAL(4,3),  -- % of events covered by patterns
    critical_event_retention DECIMAL(4,3),  -- % of critical events kept
    
    -- Compression breakdown
    compression_by_type JSONB,  -- {event_type: compression_ratio}
    patterns_used JSONB,  -- Pattern IDs used in compression
    compression_method VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_compression_date
ON event_compression_stats(date_range);
"""

# Materialized views for common queries
MATERIALIZED_VIEWS = """
-- High-performing strategies by regime
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_regime_performance AS
SELECT 
    strategy_type,
    market_regime,
    volatility_regime,
    COUNT(*) as run_count,
    AVG(sharpe_ratio) as avg_sharpe,
    AVG(total_return) as avg_return,
    STDDEV(total_return) as return_stability,
    AVG(max_drawdown) as avg_max_drawdown,
    AVG(win_rate) as avg_win_rate
FROM optimization_runs
WHERE sharpe_ratio IS NOT NULL
GROUP BY strategy_type, market_regime, volatility_regime
HAVING COUNT(*) >= 5;

CREATE INDEX idx_mv_regime_perf_sharpe 
ON mv_regime_performance(avg_sharpe DESC);

-- Pattern success by market conditions
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_pattern_conditions AS
SELECT 
    p.pattern_id,
    p.pattern_name,
    p.pattern_type,
    prp.regime_type,
    prp.success_rate,
    prp.sample_size,
    prp.optimal_threshold
FROM discovered_patterns p
JOIN pattern_regime_performance prp ON p.pattern_id = prp.pattern_id
WHERE p.status = 'active'
AND prp.sample_size >= 30;

CREATE INDEX idx_mv_pattern_cond_success
ON mv_pattern_conditions(success_rate DESC);

-- Refresh materialized views periodically
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_regime_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_pattern_conditions;
END;
$$ LANGUAGE plpgsql;
"""


def create_analytics_schema() -> List[str]:
    """
    Return all SQL statements needed to create the analytics schema.
    Execute these in order to set up the database.
    """
    return [
        OPTIMIZATION_RUNS_SCHEMA,
        TRADES_SCHEMA,
        DISCOVERED_PATTERNS_SCHEMA,
        PATTERN_PERFORMANCE_HISTORY_SCHEMA,
        PATTERN_INTERACTIONS_SCHEMA,
        PATTERN_REGIME_PERFORMANCE_SCHEMA,
        MARKET_CONDITIONS_SCHEMA,
        PERFORMANCE_ATTRIBUTION_SCHEMA,
        EVENT_COMPRESSION_STATS_SCHEMA,
        MATERIALIZED_VIEWS
    ]


# Python dataclasses for type-safe interaction with the schema
@dataclass
class OptimizationRun:
    """Represents an optimization run in the database."""
    run_id: str
    correlation_id: str
    strategy_type: str
    parameters: Dict[str, Any]
    
    # Performance metrics
    total_return: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
    profit_factor: Optional[Decimal] = None
    
    # Risk metrics
    value_at_risk: Optional[Decimal] = None
    expected_shortfall: Optional[Decimal] = None
    beta: Optional[Decimal] = None
    
    # Market conditions
    market_regime: Optional[str] = None
    volatility_regime: Optional[str] = None
    avg_market_volatility: Optional[Decimal] = None
    
    # Execution stats
    total_trades: Optional[int] = None
    avg_trade_duration: Optional[timedelta] = None
    avg_slippage: Optional[Decimal] = None
    total_commission: Optional[Decimal] = None
    
    # Event metadata
    event_count: Optional[int] = None
    first_event_id: Optional[str] = None
    last_event_id: Optional[str] = None
    event_storage_path: Optional[str] = None
    
    # Timestamps
    created_at: Optional[datetime] = None
    strategy_version: Optional[str] = None
    

@dataclass
class Trade:
    """Represents a trade in the database."""
    trade_id: str
    run_id: str
    correlation_id: str
    
    # Trade details
    entry_time: datetime
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: Decimal
    position_size: int
    
    # Exit details (optional for open trades)
    exit_time: Optional[datetime] = None
    exit_price: Optional[Decimal] = None
    
    # Performance
    pnl: Optional[Decimal] = None
    pnl_percent: Optional[Decimal] = None
    
    # Market context
    entry_volatility: Optional[Decimal] = None
    exit_volatility: Optional[Decimal] = None
    entry_regime: Optional[str] = None
    exit_regime: Optional[str] = None
    
    # Event linkage
    entry_signal_event_id: Optional[str] = None
    exit_signal_event_id: Optional[str] = None
    
    # Costs
    slippage: Optional[Decimal] = None
    commission: Optional[Decimal] = None


@dataclass
class DiscoveredPattern:
    """Represents a discovered pattern in the pattern library."""
    pattern_id: str
    pattern_type: str
    pattern_name: str
    pattern_signature: Dict[str, Any]
    
    # Performance
    success_rate: Decimal
    sample_count: int
    confidence_interval: Optional[Decimal] = None
    
    # Context
    market_conditions: Optional[Dict[str, Any]] = None
    anti_pattern: bool = False
    discovery_method: Optional[str] = None
    correlation_ids: Optional[List[str]] = None
    
    # Tracking
    live_performance: Optional[Decimal] = None
    backtest_performance: Optional[Decimal] = None
    degradation_rate: Optional[Decimal] = None
    
    # Status
    status: str = "active"  # active, degraded, retired, testing
    last_validated: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass 
class PatternPerformance:
    """Represents pattern performance history."""
    pattern_id: str
    evaluation_date: datetime
    success_rate: Decimal
    sample_size: int
    
    # Performance metrics
    avg_return: Optional[Decimal] = None
    sharpe_contribution: Optional[Decimal] = None
    
    # Error rates
    false_positive_rate: Optional[Decimal] = None
    false_negative_rate: Optional[Decimal] = None
    
    # Context
    market_conditions: Optional[Dict[str, Any]] = None
    regime_distribution: Optional[Dict[str, float]] = None