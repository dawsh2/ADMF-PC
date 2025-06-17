"""
Strategy implementations for ADMF-PC.

All strategies are simple classes with no inheritance.
They implement the Strategy protocol and can be enhanced
with capabilities through ComponentFactory.

Now includes pure function strategies decorated with @strategy
for automatic discovery.
"""

# Import only what doesn't have external dependencies
# Old class-based strategies are being phased out

# Import legacy pure function strategies for discovery
from .momentum import (
    momentum_strategy, 
    dual_momentum_strategy, 
    price_momentum_strategy
)
from .mean_reversion_simple import mean_reversion_strategy
from .rsi_strategy import rsi_strategy
from .macd_strategy import macd_strategy
from .breakout_strategy import breakout_strategy
from .ma_crossover import ma_crossover_strategy
from .rsi_composite import rsi_composite_strategy
from .rsi_tuned import rsi_tuned_strategy

# Import null strategy for testing
# from .core.null_strategy import NullStrategy  # Commented out due to import error

# Import new composable indicator strategies
from . import indicators
from . import ensemble
# from .core import *  # Keep commented if still causing issues

# Lazy imports for strategies with external dependencies
def _import_mean_reversion():
    from .mean_reversion import MeanReversionStrategy
    return MeanReversionStrategy

def _import_trend_following():
    from .trend_following import TrendFollowingStrategy, create_trend_following_strategy
    return TrendFollowingStrategy, create_trend_following_strategy

def _import_arbitrage():
    from .arbitrage import ArbitrageStrategy, create_arbitrage_strategy
    return ArbitrageStrategy, create_arbitrage_strategy

def _import_market_making():
    from .market_making import MarketMakingStrategy, create_market_making_strategy
    return MarketMakingStrategy, create_market_making_strategy


def get_strategy_class(strategy_type: str):
    """
    Get strategy class by type name.
    
    Args:
        strategy_type: Strategy type like 'momentum', 'mean_reversion', etc.
        
    Returns:
        Strategy class
    """
    strategy_map = {
        # 'null': lambda: NullStrategy,  # Commented out due to import error
        'momentum': lambda: None,  # Using pure functions now
        'mean_reversion': _import_mean_reversion,
        'trend_following': lambda: _import_trend_following()[0],
        'arbitrage': lambda: _import_arbitrage()[0],
        'market_making': lambda: _import_market_making()[0]
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    strategy_class = strategy_map[strategy_type]
    
    # Handle lazy imports
    if callable(strategy_class) and strategy_type != 'momentum':
        strategy_class = strategy_class()
    
    return strategy_class


__all__ = [
    # Classes
    "get_strategy_class",
    
    # Legacy pure function strategies (decorated)
    "momentum_strategy",
    "dual_momentum_strategy",
    "price_momentum_strategy",
    "mean_reversion_strategy",
    "rsi_strategy",
    "macd_strategy",
    "breakout_strategy",
    "ma_crossover_strategy",
    "rsi_composite_strategy",
    "rsi_tuned_strategy",
    
    # Lazy imports
    "_import_mean_reversion",
    "_import_trend_following",
    "_import_arbitrage",
    "_import_market_making"
    
    # Note: New indicator strategies, ensemble strategies, and core strategies 
    # are imported via * and will be available directly
]