"""
Data mining and analytics module for ADMF-PC.
Implements the two-layer architecture combining SQL analytics with event tracing.
"""

from src.analytics.mining.storage.schemas import (
    create_analytics_schema,
    OptimizationRun,
    Trade,
    DiscoveredPattern,
    PatternPerformance
)

from src.analytics.mining.storage.connections import (
    AnalyticsDatabase
)

from src.analytics.mining.etl.event_transformer import (
    EventTransformer,
    TradeChainBuilder,
    OptimizationAnalyzer
)

__all__ = [
    # Database
    'AnalyticsDatabase',
    
    # Schema
    'create_analytics_schema',
    'OptimizationRun',
    'Trade',
    'DiscoveredPattern',
    'PatternPerformance',
    
    # ETL
    'EventTransformer',
    'TradeChainBuilder',
    'OptimizationAnalyzer',
]