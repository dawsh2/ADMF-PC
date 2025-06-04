"""
Configuration analysis and building for workflow patterns.

This module handles:
- Pattern detection based on workflow configuration
- Multi-parameter analysis
- Pattern-specific configuration building
"""

from .pattern_detector import PatternDetector
from .parameter_analysis import ParameterAnalyzer
from .config_builders import ConfigBuilder

__all__ = [
    'PatternDetector',
    'ParameterAnalyzer', 
    'ConfigBuilder'
]