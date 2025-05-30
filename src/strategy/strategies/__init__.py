"""
Strategy implementations for ADMF-PC.

All strategies are simple classes with no inheritance.
They implement the Strategy protocol and can be enhanced
with capabilities through ComponentFactory.
"""

# Import only what doesn't have external dependencies
from .momentum import MomentumStrategy, create_momentum_strategy

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


__all__ = [
    "MomentumStrategy",
    "create_momentum_strategy",
    "_import_mean_reversion",
    "_import_trend_following",
    "_import_arbitrage",
    "_import_market_making"
]