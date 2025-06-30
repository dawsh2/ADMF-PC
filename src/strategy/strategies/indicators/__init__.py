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
- divergence.py: All divergence-based strategies
"""

# Import all strategy modules to trigger decorator registration
from . import crossovers
from . import oscillators
from . import volatility
from . import trend
from . import volume
from . import structure
from . import momentum
from . import divergence
# from . import ensemble  # Commented out - broken imports
from . import bollinger_rsi_divergence_simple
from . import bollinger_rsi_confirmed
from . import bollinger_rsi_tracker_strategy
from . import bollinger_rsi_exact_pattern
from . import bollinger_rsi_final_strategy
from . import bollinger_rsi_dependent
from . import bollinger_rsi_divergence_exact
from . import bollinger_rsi_self_contained
from . import bollinger_rsi_dependent_fixed
from . import bollinger_rsi_quick_exit
from . import bollinger_rsi_immediate
from . import bollinger_rsi_zones
from . import rsi_divergence_simple
from . import bollinger_rsi_with_exits
from . import bollinger_rsi_simple_signals
from . import bollinger_rsi_true_divergence
from . import bollinger_rsi_divergence_relaxed
from . import swing_pivot_bounce_flex
from . import swing_pivot_bounce_target
from . import swing_pivot_bounce_zones

# This ensures all @strategy decorators run and register the strategies
__all__ = [
    'crossovers',
    'oscillators', 
    'volatility',
    'trend',
    'volume',
    'structure',
    'momentum',
    'divergence',
    # 'ensemble',  # Commented out - broken imports
    'bollinger_rsi_divergence_simple',
    'bollinger_rsi_confirmed',
    'bollinger_rsi_tracker_strategy',
    'bollinger_rsi_exact_pattern',
    'bollinger_rsi_final_strategy',
    'bollinger_rsi_dependent',
    'bollinger_rsi_divergence_exact',
    'bollinger_rsi_dependent_fixed',
    'bollinger_rsi_quick_exit',
    'bollinger_rsi_immediate',
    'bollinger_rsi_zones',
    'rsi_divergence_simple',
    'bollinger_rsi_with_exits',
    'bollinger_rsi_simple_signals',
    'bollinger_rsi_true_divergence',
    'bollinger_rsi_divergence_relaxed',
    'swing_pivot_bounce_flex',
    'swing_pivot_bounce_target',
    'swing_pivot_bounce_zones'
]