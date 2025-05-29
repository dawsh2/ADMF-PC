"""
Market making strategy placeholder.
"""

from ..base import StrategyBase


class MarketMakingStrategy(StrategyBase):
    """Market making strategy - to be implemented."""
    
    def __init__(self, name: str = "market_making_strategy"):
        super().__init__(name)
        # TODO: Implement market making logic
    
    def on_bar(self, event):
        """Process bar event."""
        pass