"""
Data mining and pattern discovery for optimization analysis.

Implements two-layer architecture:
- SQL database for high-level metrics and quick queries
- Event traces for detailed pattern analysis
"""

from .models import OptimizationRun
from .protocols import MetricsAggregatorProtocol
from .two_layer_mining import TwoLayerMiningSystem
from .pattern_discovery import TradingPatternMiner
from .query import EventQueryInterface

# Aliases for backward compatibility
TwoLayerMiningArchitecture = TwoLayerMiningSystem
PatternMiner = TradingPatternMiner

__all__ = [
    # Models
    'OptimizationRun',
    
    # Protocols
    'MetricsAggregatorProtocol',
    
    # Core components
    'TwoLayerMiningSystem',
    'TradingPatternMiner', 
    'EventQueryInterface',
    
    # Backward compatibility aliases
    'TwoLayerMiningArchitecture',
    'PatternMiner'
]