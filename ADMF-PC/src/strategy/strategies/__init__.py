"""
Strategy implementations for ADMF-PC.

All strategies are simple classes with no inheritance.
They implement the Strategy protocol and can be enhanced
with capabilities through ComponentFactory.
"""

from .momentum import MomentumStrategy, create_momentum_strategy
from .mean_reversion import MeanReversionStrategy
from .trend_following import TrendFollowingStrategy, create_trend_following_strategy
from .arbitrage import ArbitrageStrategy, create_arbitrage_strategy
from .market_making import MarketMakingStrategy, create_market_making_strategy


__all__ = [
    "MomentumStrategy",
    "create_momentum_strategy",
    "MeanReversionStrategy",
    "TrendFollowingStrategy",
    "create_trend_following_strategy",
    "ArbitrageStrategy",
    "create_arbitrage_strategy",
    "MarketMakingStrategy",
    "create_market_making_strategy"
]