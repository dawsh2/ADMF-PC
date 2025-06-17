"""
Indicator-based Strategy Implementations.

All strategies in this module are automatically discovered via the @strategy decorator.
The modules must be imported for the decorators to run and register the strategies.

Organized by indicator category:
- crossovers.py: All crossover-based strategies
- oscillators.py: All oscillator-based strategies
- volatility.py: All volatility-based strategies
- trend.py: All trend-based strategies
- volume.py: All volume-based strategies
- structure.py: All structure-based strategies
- momentum.py: All momentum-based strategies
"""

# Import all strategy modules to trigger decorator registration
from . import crossovers
from . import oscillators
from . import volatility
from . import trend
from . import volume
from . import structure
from . import momentum

# This ensures all @strategy decorators run and register the strategies
__all__ = [
    'crossovers',
    'oscillators', 
    'volatility',
    'trend',
    'volume',
    'structure',
    'momentum'
]