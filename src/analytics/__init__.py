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

__all__ = [
    "BacktestReportGenerator",
    "AnalyticsDatabase", 
    "EventTransformer",
    "TradeChainBuilder",
    "OptimizationAnalyzer",
]