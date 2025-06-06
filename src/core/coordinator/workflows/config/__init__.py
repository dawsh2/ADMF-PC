"""
Configuration analysis and building for unified architecture.

This module handles:
- Multi-parameter analysis (still useful for parameter grids)
- Configuration building helpers
"""

# Pattern detector removed - unified architecture doesn't need it!
# from .pattern_detector import PatternDetector
from .parameter_analysis import ParameterAnalyzer
from .config_builders import ConfigBuilder

__all__ = [
    # 'PatternDetector',  # REMOVED - no pattern detection in unified architecture
    'ParameterAnalyzer', 
    'ConfigBuilder'
]