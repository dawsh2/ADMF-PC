"""
ADMF-PC Analytics - Simple SQL-based trace analysis
"""

from .simple_analytics import TraceAnalysis, quick_analysis, QUERIES

__version__ = "1.0.0"

__all__ = [
    'TraceAnalysis',
    'quick_analysis',
    'QUERIES'
]