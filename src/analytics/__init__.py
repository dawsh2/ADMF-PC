"""
Minimal Parallelization-Ready Analytics Module

Provides immediate value for optimization analysis while maintaining
a clean foundation for future sophisticated features.

Key Features:
- SQL-first pattern discovery
- Correlation ID bridge to event traces  
- Container-isolated parallel execution
- Protocol-based composition architecture
"""

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

__all__ = [
    'ContainerMetricsExtractor',
    'SimplePatternDetector', 
    'MinimalReportGenerator',
    # Mining
    'TwoLayerMiningArchitecture',
    'PatternMiner',
    'OptimizationRun',
    'MetricsAggregatorProtocol'
]