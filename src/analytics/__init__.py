"""
ADMF-PC Analytics Module

Universal SQL-based analytics interface for trading system analysis.
Provides scalable analysis of optimization results, strategy performance,
and regime-aware trading insights.

Key Features:
- SQL-first analytics interface (universal, no learning curve)
- Hybrid SQL + Parquet storage (metadata + sparse signals)
- Interactive exploration and programmatic access
- Scalable to thousands of strategies
- Custom trading-specific SQL functions

Main Entry Point:
    from analytics import AnalyticsWorkspace
    
    workspace = AnalyticsWorkspace('path/to/workspace')
    results = workspace.sql("SELECT * FROM strategies WHERE sharpe_ratio > 1.5")
"""

# New SQL-based interface (primary)
from .workspace import AnalyticsWorkspace
from .exceptions import AnalyticsError, WorkspaceNotFoundError, QueryError
from .migration import migrate_workspace, setup_workspace, find_workspaces, WorkspaceMigrator
from .integration import integrate_with_topology_result, get_analytics_integrator, AnalyticsIntegrator

# Legacy components (for backward compatibility)
from .metrics import ContainerMetricsExtractor
from .patterns import SimplePatternDetector
from .reports import MinimalReportGenerator

# Data mining features
from .mining import (
    TwoLayerMiningArchitecture,
    PatternMiner,
    OptimizationRun,
    MetricsAggregatorProtocol
)

# Strategy filtering and analysis
from .strategy_filter import StrategyFilter, analyze_grid_search

__version__ = "2.0.0"

__all__ = [
    # Primary SQL interface
    'AnalyticsWorkspace',
    'AnalyticsError',
    'WorkspaceNotFoundError', 
    'QueryError',
    
    # Migration utilities
    'migrate_workspace',
    'setup_workspace',
    'find_workspaces',
    'WorkspaceMigrator',
    
    # Integration utilities
    'integrate_with_topology_result',
    'get_analytics_integrator',
    'AnalyticsIntegrator',
    
    # Legacy components
    'ContainerMetricsExtractor',
    'SimplePatternDetector', 
    'MinimalReportGenerator',
    
    # Mining
    'TwoLayerMiningArchitecture',
    'PatternMiner',
    'OptimizationRun',
    'MetricsAggregatorProtocol',
    
    # Strategy filtering
    'StrategyFilter',
    'analyze_grid_search'
]