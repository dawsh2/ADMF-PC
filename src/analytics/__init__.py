"""
Analytics module for ADMF-PC.

This module encompasses all analysis capabilities including:
- Data mining and pattern discovery
- Performance analytics and metrics
- Report generation and visualization
- Event trace analysis
"""
from src.analytics.basic_report import BacktestReportGenerator
from src.analytics.mining import (
    AnalyticsDatabase,
    EventTransformer,
    TradeChainBuilder,
    OptimizationAnalyzer,
)
from src.analytics.trace_analysis import (
    TraceAnalyzer,
    TraceArchiver,
    analyze_trace,
    EventExtractor,
    PortfolioMetricsExtractor,
    SignalExtractor,
    TradeExtractor,
)
from src.analytics.metrics_collection import (
    MetricsCollector,
    get_phase_metrics,
)

__all__ = [
    "BacktestReportGenerator",
    "AnalyticsDatabase", 
    "EventTransformer",
    "TradeChainBuilder",
    "OptimizationAnalyzer",
    "TraceAnalyzer",
    "TraceArchiver",
    "analyze_trace",
    "EventExtractor",
    "PortfolioMetricsExtractor",
    "SignalExtractor",
    "TradeExtractor",
    "MetricsCollector",
    "get_phase_metrics",
]