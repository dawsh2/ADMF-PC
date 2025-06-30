"""
Analytics Modules - Organized, reusable analysis components

Usage:
    from analytics.modules.trade_analysis import extract_trades, filter_by_frequency
    from analytics.modules.regime_analysis import add_volatility_regime, add_trend_regime
    from analytics.modules.optimization import optimize_stops_by_regime
"""

# Convenient imports
from . import core
from . import trade_analysis
from . import regime_analysis
from . import optimization
from . import performance
from . import visualization

__version__ = "0.1.0"